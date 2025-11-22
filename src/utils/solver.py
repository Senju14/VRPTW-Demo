"""
VRPTW Solver using OR-Tools.
"""

from typing import List, Tuple, Dict, Any
import time
from src.utils.data_loader import Customer


def solve_vrptw(depot: Customer, customers: List[Customer], vehicle_capacity: int, num_vehicles: int) -> Dict[str, Any]:
    """
    Solve VRPTW using OR-Tools.

    Args:
        depot: Depot customer.
        customers: List of customers.
        vehicle_capacity: Capacity of each vehicle.
        num_vehicles: Number of vehicles.

    Returns:
        Dict with routes, total_distance, execution_time, violations.
    """
    try:
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
    except ImportError:
        raise ImportError("ortools not installed. Please install with: uv add ortools")

    start_time = time.time()

    # Create data model
    data = create_data_model(depot, customers, vehicle_capacity, num_vehicles)

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        data['vehicle_capacities'][0] // 10,  # allow some capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Add Time Window constraint
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    
    # Calculate max time from data
    max_time = max([tw[1] for tw in data['time_windows']]) + 1000
    
    routing.AddDimension(
        time_callback_index,
        max_time,  # allow waiting time
        max_time,  # maximum time per vehicle
        False,  # Don't force start cumul to zero.
        'Time')

    time_dimension = routing.GetDimensionOrDie('Time')
    for location_idx, time_window in enumerate(data['time_windows']):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    # Instantiate route start and end times to produce feasible times.
    for i in range(data['num_vehicles']):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i)))
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(i)))

    # Allow dropping nodes if solution is infeasible
    penalty = 1000000
    for node in range(1, len(data['time_windows'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    
    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
    search_parameters.time_limit.seconds = 60  # Increase time limit
    search_parameters.solution_limit = 100

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    execution_time = time.time() - start_time

    if solution:
        routes = extract_routes(manager, routing, solution)
        total_distance = solution.ObjectiveValue()
        violations = check_violations(routes, data)
        return {
            'routes': routes,
            'total_distance': total_distance,
            'execution_time': execution_time,
            'violations': violations
        }
    else:
        return {
            'routes': [],
            'total_distance': 0,
            'execution_time': execution_time,
            'violations': ['No solution found']
        }


def create_data_model(depot: Customer, customers: List[Customer], vehicle_capacity: int, num_vehicles: int) -> Dict:
    """
    Create data model for OR-Tools.
    """
    all_customers = [depot] + customers
    locations = [(c.x, c.y) for c in all_customers]
    demands = [c.demand for c in all_customers]
    time_windows = [(c.ready_time, c.due_date) for c in all_customers]

    # Distance matrix
    distance_matrix = []
    for i in range(len(locations)):
        row = []
        for j in range(len(locations)):
            dist = ((locations[i][0] - locations[j][0])**2 + (locations[i][1] - locations[j][1])**2)**0.5
            row.append(int(dist))  # integer distances
        distance_matrix.append(row)

    # Time matrix (assume speed 1, so time = distance)
    time_matrix = distance_matrix

    return {
        'locations': locations,
        'distance_matrix': distance_matrix,
        'time_matrix': time_matrix,
        'demands': demands,
        'time_windows': time_windows,
        'num_vehicles': num_vehicles,
        'depot': 0,
        'vehicle_capacities': [vehicle_capacity] * num_vehicles
    }


def extract_routes(manager, routing, solution) -> List[List[int]]:
    """
    Extract routes from solution.
    """
    routes = []
    for vehicle_id in range(routing.vehicles()):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            index = solution.Value(routing.NextVar(index))
        routes.append(route)
    return routes


def check_violations(routes: List[List[int]], data: Dict) -> List[str]:
    """
    Check for violations (simplified).
    """
    violations = []
    # For now, assume no violations if solution found
    return violations