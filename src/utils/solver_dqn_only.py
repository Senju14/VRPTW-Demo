"""
Pure DQN solver for VRPTW.
"""

import time
import random
import copy

from src.utils.solver_common import (
    create_initial_solution,
    evaluate_solution,
    random_removal,
    route_removal,
    greedy_insertion,
    DQNAgent,
    get_state,
)


def solve_dqn_only_vrptw(depot, customers, capacity, num_vehicles, model_path, iterations=300):
    """Solve VRPTW using DQN only (simpler than DQN-ALNS)."""
    start_time = time.time()

    print(f"[DQN-ONLY] Starting with {len(customers)} customers, {num_vehicles} vehicles")

    destroy_operators = [random_removal, route_removal]
    agent = DQNAgent(state_size=2, action_size=2)
    agent.load_weights(model_path)
    print(f"[DQN-ONLY] Loaded model: {model_path.split('/')[-1]}")

    initial_solution = create_initial_solution(depot, customers, capacity, num_vehicles)
    current_solution = copy.deepcopy(initial_solution)
    best_solution = copy.deepcopy(initial_solution)
    iterations_since_best = 0

    print(
        f"[DQN-ONLY] Initial: cost={best_solution.total_cost:.1f}, "
        f"vehicles={len(best_solution.routes)}/{num_vehicles}"
    )

    for iteration in range(1, iterations + 1):
        state = get_state(current_solution.total_cost, best_solution.total_cost, iterations_since_best)
        action_idx = agent.select_action(state, epsilon=0.0)
        destroy_op = destroy_operators[action_idx]

        new_solution = copy.deepcopy(current_solution)

        if destroy_op == random_removal:
            num_to_remove = random.randint(5, 15)
            removed = destroy_op(new_solution, depot, capacity, num_to_remove)
        else:
            max_routes = min(len(new_solution.routes), max(1, len(new_solution.routes) // 3))
            num_routes = random.randint(1, max_routes)
            removed = destroy_op(new_solution, depot, capacity, num_routes)

        greedy_insertion(new_solution, removed, depot, capacity, num_vehicles)
        evaluate_solution(new_solution, depot, capacity)

        if len(new_solution.routes) <= num_vehicles:
            if new_solution.total_cost < best_solution.total_cost:
                best_solution = copy.deepcopy(new_solution)
                current_solution = copy.deepcopy(new_solution)
                iterations_since_best = 0
            else:
                iterations_since_best += 1
                if random.random() < 0.05:
                    current_solution = copy.deepcopy(new_solution)
        else:
            iterations_since_best += 1

        if iteration % 100 == 0:
            print(f"  Progress: {iteration}/{iterations}, best_cost={best_solution.total_cost:.1f}")

    execution_time = time.time() - start_time

    routes_as_indices = []
    for route in best_solution.routes:
        route_indices = [0]
        for customer in route:
            route_indices.append(customer.id)
        routes_as_indices.append(route_indices)

    print(f"[DQN-ONLY] Completed in {execution_time:.1f}s")
    print(
        f"[DQN-ONLY] Final: cost={best_solution.total_cost:.1f}, "
        f"vehicles={len(best_solution.routes)}/{num_vehicles}"
    )

    return {
        "routes": routes_as_indices,
        "total_distance": best_solution.total_cost,
        "execution_time": execution_time,
        "violations": [],
    }


__all__ = ["solve_dqn_only_vrptw"]


