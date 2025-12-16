"""
OR-Tools baseline solver for VRPTW.
"""
import math
import time
from typing import Dict, List, Any

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.backend.data_loader import Customer


def solve_vrptw(
    depot: Customer,
    customers: List[Customer],
    capacity: int,
    num_vehicles: int,
) -> Dict[str, Any]:
    """Solve VRPTW using OR-Tools."""
    start = time.time()
    all_customers = [depot] + customers
    locs = [(c.x, c.y) for c in all_customers]
    demands = [c.demand for c in all_customers]
    tw = [(c.ready_time, c.due_date) for c in all_customers]

    dist: List[List[int]] = []
    for i in range(len(locs)):
        row: List[int] = []
        for j in range(len(locs)):
            d = math.hypot(locs[i][0] - locs[j][0], locs[i][1] - locs[j][1])
            row.append(int(d))
        dist.append(row)

    time_matrix = dist
    manager = pywrapcp.RoutingIndexManager(len(locs), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(from_index, to_index):
        a = manager.IndexToNode(from_index)
        b = manager.IndexToNode(to_index)
        return dist[a][b]

    transit_index = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    def demand_cb(from_index):
        a = manager.IndexToNode(from_index)
        return demands[a]

    demand_index = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
        capacity // 10,
        [capacity] * num_vehicles,
        True,
        "Capacity",
    )

    def time_cb(from_index, to_index):
        a = manager.IndexToNode(from_index)
        b = manager.IndexToNode(to_index)
        return time_matrix[a][b]

    time_index = routing.RegisterTransitCallback(time_cb)
    max_time = max(b for _, b in tw) + 1000
    routing.AddDimension(
        time_index,
        max_time,
        max_time,
        False,
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    for i, (a, b) in enumerate(tw):
        if i == 0:
            continue
        idx = manager.NodeToIndex(i)
        time_dim.CumulVar(idx).SetRange(a, b)

    for v in range(num_vehicles):
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.Start(v)))
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    penalty = 1_000_000
    for node in range(1, len(tw)):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
    params.time_limit.seconds = 60
    params.solution_limit = 100

    solution = routing.SolveWithParameters(params)
    elapsed = time.time() - start

    if not solution:
        return {
            "routes": [],
            "total_distance": 0,
            "execution_time": elapsed,
            "violations": ["no_solution"],
        }

    routes: List[List[int]] = []
    for v in range(num_vehicles):
        idx = routing.Start(v)
        r: List[int] = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            r.append(node)
            idx = solution.Value(routing.NextVar(idx))
        if r:
            routes.append(r)

    total_distance = solution.ObjectiveValue()
    return {
        "routes": routes,
        "total_distance": total_distance,
        "execution_time": elapsed,
        "violations": [],
    }

