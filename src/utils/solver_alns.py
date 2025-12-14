"""
Pure ALNS solver for VRPTW.
"""

import time
import math
import random
import copy

from src.utils.solver_common import (
    create_initial_solution,
    evaluate_solution,
    random_removal,
    route_removal,
    greedy_insertion,
)


def select_operator_roulette(weights):
    """Roulette wheel selection for ALNS."""
    total = sum(weights)
    if total == 0:
        return random.randint(0, len(weights) - 1)
    r = random.uniform(0, total)
    cumsum = 0
    for i, w in enumerate(weights):
        cumsum += w
        if r <= cumsum:
            return i
    return len(weights) - 1


def solve_alns_vrptw(depot, customers, capacity, num_vehicles, iterations=300):
    """Solve VRPTW using pure ALNS (roulette wheel selection)."""
    start_time = time.time()

    print(f"[ALNS] Starting with {len(customers)} customers, {num_vehicles} vehicles")

    destroy_operators = [random_removal, route_removal]
    destroy_weights = [1.0] * len(destroy_operators)
    destroy_scores = [0.0] * len(destroy_operators)
    destroy_uses = [0] * len(destroy_operators)
    segment_length = 100

    R1, R2, R3 = 5, 2, 1  # Global best, local improvement, accept worse
    reaction_factor = 0.1
    start_temp, end_temp = 100.0, 0.1

    initial_solution = create_initial_solution(depot, customers, capacity, num_vehicles)
    current_solution = copy.deepcopy(initial_solution)
    best_solution = copy.deepcopy(initial_solution)

    print(f"[ALNS] Initial: cost={best_solution.total_cost:.1f}, vehicles={len(best_solution.routes)}/{num_vehicles}")

    for iteration in range(1, iterations + 1):
        new_solution = copy.deepcopy(current_solution)

        d_idx = select_operator_roulette(destroy_weights)
        destroy_op = destroy_operators[d_idx]

        if destroy_op == random_removal:
            num_to_remove = random.randint(5, 15)
            removed = destroy_op(new_solution, depot, capacity, num_to_remove)
        else:
            max_routes = min(len(new_solution.routes), max(1, len(new_solution.routes) // 3))
            num_routes = random.randint(1, max_routes)
            removed = destroy_op(new_solution, depot, capacity, num_routes)

        greedy_insertion(new_solution, removed, depot, capacity, num_vehicles)
        evaluate_solution(new_solution, depot, capacity)

        score = 0
        if len(new_solution.routes) <= num_vehicles:
            if new_solution.total_cost < best_solution.total_cost:
                best_solution = copy.deepcopy(new_solution)
                current_solution = copy.deepcopy(new_solution)
                score = R1
            elif new_solution.total_cost < current_solution.total_cost:
                current_solution = copy.deepcopy(new_solution)
                score = R2
            else:
                delta = new_solution.total_cost - current_solution.total_cost
                temp = start_temp * (end_temp / start_temp) ** (iteration / iterations)
                if math.exp(-delta / temp) > random.random():
                    current_solution = copy.deepcopy(new_solution)
                    score = R3

        destroy_scores[d_idx] += score
        destroy_uses[d_idx] += 1

        if iteration % segment_length == 0:
            for i in range(len(destroy_operators)):
                if destroy_uses[i] > 0:
                    destroy_weights[i] = destroy_weights[i] * (1 - reaction_factor) + reaction_factor * (
                        destroy_scores[i] / destroy_uses[i]
                    )
                    destroy_scores[i] = 0
                    destroy_uses[i] = 0

        if iteration % 100 == 0:
            print(f"  Progress: {iteration}/{iterations}, best_cost={best_solution.total_cost:.1f}")

    execution_time = time.time() - start_time

    routes_as_indices = []
    for route in best_solution.routes:
        route_indices = [0]
        for customer in route:
            route_indices.append(customer.id)
        routes_as_indices.append(route_indices)

    print(f"[ALNS] Completed in {execution_time:.1f}s")
    print(f"[ALNS] Final: cost={best_solution.total_cost:.1f}, vehicles={len(best_solution.routes)}/{num_vehicles}")

    return {
        "routes": routes_as_indices,
        "total_distance": best_solution.total_cost,
        "execution_time": execution_time,
        "violations": [],
    }


__all__ = ["solve_alns_vrptw"]


