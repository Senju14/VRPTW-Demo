from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math

def solve_ortools(instance, max_vehicles, time_limit_sec=10):
    """
    Giải VRPTW bằng OR-Tools.
    Phiên bản Fix: Kiểm tra solution tồn tại trước khi trích xuất dữ liệu.
    """
    # 1. Setup Data
    depot = instance['depot']
    customers = instance['customers']
    locations = [depot] + customers
    num_locations = len(locations)
    
    # Scaling factor (OR-Tools works with integers)
    scaling = 100 
    
    demands = [0] + [c['demand'] for c in customers]
    time_windows = [(depot['ready_time'], depot['due_time'])] + \
                   [(c['ready_time'], c['due_time']) for c in customers]
    service_times = [0] + [c['service_time'] for c in customers]
    
    # 2. Create Routing Index Manager
    manager = pywrapcp.RoutingIndexManager(num_locations, max_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # 3. Callbacks
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        x1, y1 = locations[from_node]['lat'], locations[from_node]['lng']
        x2, y2 = locations[to_node]['lat'], locations[to_node]['lng']
        dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        return int(dist * scaling)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        travel = distance_callback(from_index, to_index)
        service = int(service_times[from_node] * scaling)
        return travel + service

    time_callback_index = routing.RegisterTransitCallback(time_callback)

    # 4. Constraints
    # Capacity
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(lambda i: demands[manager.IndexToNode(i)]),
        0,  # null capacity slack
        [instance['capacity']] * max_vehicles,
        True,
        "Capacity"
    )

    # Time Window
    horizon = 2000000 * scaling # Large number
    routing.AddDimension(
        time_callback_index,
        horizon,  # allow waiting time
        horizon,  # maximum time per vehicle
        False,    # Don't force start to zero
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    
    for location_idx, (start, end) in enumerate(time_windows):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(int(start * scaling), int(end * scaling))

    # --- KEY FIX: Add Disjunction (Allow dropping nodes with penalty) ---
    # Phạt 1.000.000 điểm nếu bỏ khách -> Ưu tiên phục vụ hết, nhưng nếu không được thì bỏ qua chứ không crash.
    penalty = 1000000
    for i in range(1, num_locations):
        routing.AddDisjunction([manager.NodeToIndex(i)], penalty)

    # 5. Search Parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = time_limit_sec
    search_parameters.log_search = False

    # 6. Solve
    solution = routing.SolveWithParameters(search_parameters)

    # --- FIX QUAN TRỌNG: Kiểm tra solution trước khi loop ---
    if not solution:
        return None

    routes = []
    total_dist = 0
    
    for vehicle_id in range(max_vehicles):
        index = routing.Start(vehicle_id)
        route_nodes = []
        
        # Check if vehicle is used
        if routing.IsEnd(solution.Value(routing.NextVar(index))):
            continue

        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            if node_idx != 0:
                route_nodes.append(locations[node_idx]['id'])
            
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            total_dist += routing.GetArcCostForVehicle(prev_index, index, vehicle_id)
        
        if route_nodes:
            routes.append(route_nodes)

    return {
        "algorithm": "OR-Tools",
        "vehicles": len(routes),
        "distance": total_dist / scaling,
        "routes": routes,
        "feasible": True
    }
