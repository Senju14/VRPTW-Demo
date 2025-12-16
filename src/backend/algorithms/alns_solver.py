"""
ALNS (Adaptive Large Neighborhood Search) solver for VRPTW.
"""
import copy
import math
import random
import time
from typing import Dict, List, Any

from src.backend.data_loader import Customer
from src.backend.solver import (
    Solution,
    create_initial_solution,
    evaluate_solution,
    random_removal,
    route_removal,
    greedy_insertion,
    routes_to_indices,
)


def solve_alns_vrptw(
    depot: Customer,
    customers: List[Customer],
    capacity: int,
    num_vehicles: int,
    iterations: int = 300,
) -> Dict[str, Any]:
    """Solve VRPTW using ALNS."""
    destroy_ops = [random_removal, route_removal]
    weights = [1.0] * len(destroy_ops)
    scores = [0.0] * len(destroy_ops)
    uses = [0] * len(destroy_ops)
    seg = 100
    R1, R2, R3 = 5, 2, 1
    alpha = 0.1
    t0, t1 = 100.0, 0.1

    best = create_initial_solution(depot, customers, capacity, num_vehicles)
    cur = copy.deepcopy(best)
    start = time.time()

    def pick(ws):
        s = sum(ws)
        if s == 0:
            return random.randint(0, len(ws) - 1)
        r = random.uniform(0, s)
        c = 0
        for i, w in enumerate(ws):
            c += w
            if r <= c:
                return i
        return len(ws) - 1

    for it in range(1, iterations + 1):
        new = copy.deepcopy(cur)
        d_idx = pick(weights)
        op = destroy_ops[d_idx]

        if op is random_removal:
            k = random.randint(5, 15)
            removed = op(new, depot, capacity, k)
        else:
            m = min(len(new.routes), max(1, len(new.routes) // 3))
            k = random.randint(1, m)
            removed = op(new, depot, capacity, k)

        greedy_insertion(new, removed, depot, capacity, num_vehicles)
        evaluate_solution(new, depot, capacity)

        score = 0
        if len(new.routes) <= num_vehicles:
            if new.total_cost < best.total_cost:
                best = copy.deepcopy(new)
                cur = copy.deepcopy(new)
                score = R1
            elif new.total_cost < cur.total_cost:
                cur = copy.deepcopy(new)
                score = R2
            else:
                delta = new.total_cost - cur.total_cost
                temp = t0 * (t1 / t0) ** (it / iterations)
                if math.exp(-delta / temp) > random.random():
                    cur = copy.deepcopy(new)
                    score = R3

        scores[d_idx] += score
        uses[d_idx] += 1

        if it % seg == 0:
            for i in range(len(destroy_ops)):
                if uses[i]:
                    w = scores[i] / uses[i]
                    weights[i] = weights[i] * (1 - alpha) + alpha * w
                    scores[i] = 0
                    uses[i] = 0

    elapsed = time.time() - start
    routes = routes_to_indices(best)
    return {
        "routes": routes,
        "total_distance": best.total_cost,
        "execution_time": elapsed,
        "violations": [],
    }

