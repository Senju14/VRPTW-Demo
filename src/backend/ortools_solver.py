def solve_with_ortools(instance, max_vehicles):
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    from .utils import Solution
    
    manager = pywrapcp.RoutingIndexManager(len(instance.coords), max_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(instance.distance(from_node, to_node) * 100)
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    def demand_callback(from_idx):
        from_node = manager.IndexToNode(from_idx)
        return int(instance.demands[from_node])
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, 
        [int(instance.capacity)] * max_vehicles,
        True, 'Capacity')
    
    def time_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        travel = int(instance.distance(from_node, to_node) * 100)
        service = int(instance.service_times[from_node] * 100)
        return travel + service
    
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_callback_index, 30000, 300000, False, 'Time')
    time_dimension = routing.GetDimensionOrDie('Time')
    
    for node in range(1, len(instance.coords)):
        index = manager.NodeToIndex(node)
        time_dimension.CumulVar(index).SetRange(
            int(instance.ready_times[node] * 100),
            int(instance.due_times[node] * 100))
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 30
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        raise Exception("OR-Tools couldn't find a solution")
    
    routes = []
    for vehicle_id in range(max_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                route.append(node)
            index = solution.Value(routing.NextVar(index))
        if route:
            routes.append(route)
    
    return Solution(routes, instance)

