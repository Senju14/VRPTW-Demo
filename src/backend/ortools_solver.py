from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math

def solve_ortools(instance, max_vehicles, time_limit_sec=30):
    """
    Giải bài toán VRPTW sử dụng Google OR-Tools.
    Tuân thủ ràng buộc: Số xe, Sức chứa, Khung thời gian.
    """
    # 1. Chuẩn bị dữ liệu
    depot = instance['depot']
    customers = instance['customers']
    locations = [depot] + customers
    
    num_locations = len(locations)
    demands = [0] + [c['demand'] for c in customers]
    time_windows = [(depot['ready_time'], depot['due_time'])] + \
                   [(c['ready_time'], c['due_time']) for c in customers]
    service_times = [0] + [c['service_time'] for c in customers]
    
    # 2. Tạo Manager
    manager = pywrapcp.RoutingIndexManager(
        num_locations, 
        max_vehicles, 
        0 # Depot index
    )
    routing = pywrapcp.RoutingModel(manager)

    # 3. Callback khoảng cách (Euclidean)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        x1, y1 = locations[from_node]['lat'], locations[from_node]['lng']
        x2, y2 = locations[to_node]['lat'], locations[to_node]['lng']
        dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        return int(dist * 100) # OR-Tools dùng số nguyên

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 4. Callback thời gian (Distance + Service Time)
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_callback(from_index, to_index) + service_times[from_node] * 100

    time_callback_index = routing.RegisterTransitCallback(time_callback)

    # 5. Ràng buộc Capacity
    routing.AddDimension(
        transit_callback_index,
        0,  # null capacity slack
        instance['capacity'] * 1000, # Max capacity (cho lỏng lẻo nếu cần)
        True,  # start cumul to zero
        "Distance"
    )
    
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(lambda i: demands[manager.IndexToNode(i)]),
        0,  # null capacity slack
        [instance['capacity']] * max_vehicles,  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity"
    )

    # 6. Ràng buộc Time Windows
    routing.AddDimension(
        time_callback_index,
        300000,  # allow waiting time (Horizon lớn)
        300000,  # maximum time per vehicle
        False, # Don't force start cumul to zero
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    for location_idx, (start, end) in enumerate(time_windows):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(start * 100, end * 100)

    # 7. Tham số Search
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.time_limit.seconds = time_limit_sec

    # 8. Giải
    solution = routing.SolveWithParameters(search_parameters)

    # 9. Format kết quả trả về
    if solution:
        routes = []
        total_dist = 0
        total_time = 0
        
        for vehicle_id in range(max_vehicles):
            index = routing.Start(vehicle_id)
            route_nodes = []
            
            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                continue # Xe không dùng
                
            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                if node_idx != 0: # Bỏ qua depot trong danh sách nodes của route
                    route_nodes.append(locations[node_idx])
                
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                total_dist += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            
            if route_nodes:
                routes.append({"nodes": route_nodes})

        return {
            "algorithm": "OR-Tools",
            "vehicles": len(routes),
            "distance": total_dist / 100.0,
            "time": 0.0, # Placeholder
            "routes": routes,
            "depot": depot
        }
    return None