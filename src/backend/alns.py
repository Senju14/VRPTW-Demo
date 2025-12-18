import numpy as np
import random
from typing import List
from .utils import VRPTWInstance, Solution, compute_route_metrics

class ALNS:
    def __init__(self, instance: VRPTWInstance, max_iterations=1000,
                 destroy_size=0.3, temp_start=100, temp_decay=0.99):
        self.instance = instance
        self.max_iterations = max_iterations
        self.destroy_size = max(1, int(destroy_size * instance.n))
        self.temp = temp_start
        self.temp_decay = temp_decay

        self.destroy_ops = [self.random_removal, self.worst_removal, self.shaw_removal]
        self.repair_ops = [self.greedy_insert, self.regret_insert]

        self.destroy_weights = np.ones(len(self.destroy_ops))
        self.repair_weights = np.ones(len(self.repair_ops))

    def solve(self, initial_solution: Solution = None):
        if initial_solution is None:
            current = self._create_initial_solution()
        else:
            current = initial_solution.copy()

        best = current.copy()

        for iteration in range(self.max_iterations):
            destroy_idx = self._select_operator(self.destroy_weights)
            repair_idx = self._select_operator(self.repair_weights)

            removed = self.destroy_ops[destroy_idx](current.copy())
            candidate = self.repair_ops[repair_idx](removed)

            if self._accept(current, candidate):
                current = candidate
                if candidate.feasible and candidate.cost < best.cost:
                    best = candidate.copy()

            self.temp *= self.temp_decay

        return best

    def _create_initial_solution(self):
        inst = self.instance
        unvisited = list(range(1, inst.n + 1))
        routes = []

        while unvisited:
            route = []
            load = 0
            time = 0
            current = 0

            while unvisited:
                feasible_nodes = []
                for node in unvisited:
                    if load + inst.demands[node] <= inst.capacity:
                        arrival = time + inst.distance(current, node)
                        if arrival <= inst.due_times[node]:
                            feasible_nodes.append(node)

                if not feasible_nodes:
                    break

                next_node = min(feasible_nodes, key=lambda n: inst.distance(current, n))
                route.append(next_node)
                unvisited.remove(next_node)

                load += inst.demands[next_node]
                time += inst.distance(current, next_node)
                time = max(time, inst.ready_times[next_node])
                time += inst.service_times[next_node]
                current = next_node

            if route:
                routes.append(route)

        return Solution(routes, inst)

    def _select_operator(self, weights):
        probs = weights / weights.sum()
        return np.random.choice(len(weights), p=probs)

    def _accept(self, current: Solution, candidate: Solution):
        if not candidate.feasible:
            return False
        if candidate.cost < current.cost:
            return True
        delta = candidate.cost - current.cost
        return random.random() < np.exp(-delta / max(self.temp, 1e-5))

    def random_removal(self, solution: Solution):
        nodes_to_remove = random.sample(range(1, self.instance.n + 1),
                                      min(self.destroy_size, self.instance.n))
        removed = []
        for i, route in enumerate(solution.routes):
            solution.routes[i] = [n for n in route if n not in nodes_to_remove]
        solution.routes = [r for r in solution.routes if r]
        return solution, nodes_to_remove

    def worst_removal(self, solution: Solution):
        costs = []
        for route in solution.routes:
            for i, node in enumerate(route):
                prev = route[i-1] if i > 0 else 0
                next_node = route[i+1] if i < len(route)-1 else 0
                cost_before = self.instance.distance(prev, node) + self.instance.distance(node, next_node)
                cost_after = self.instance.distance(prev, next_node)
                costs.append((cost_before - cost_after, node))

        costs.sort(reverse=True)
        nodes_to_remove = [node for _, node in costs[:self.destroy_size]]

        for i, route in enumerate(solution.routes):
            solution.routes[i] = [n for n in route if n not in nodes_to_remove]
        solution.routes = [r for r in solution.routes if r]
        return solution, nodes_to_remove

    def shaw_removal(self, solution: Solution):
        all_nodes = [n for route in solution.routes for n in route]
        if not all_nodes:
            return solution, []

        seed = random.choice(all_nodes)
        nodes_to_remove = [seed]

        while len(nodes_to_remove) < self.destroy_size and len(nodes_to_remove) < len(all_nodes):
            relatedness = []
            for node in all_nodes:
                if node not in nodes_to_remove:
                    dist = self.instance.distance(seed, node)
                    time_diff = abs(self.instance.ready_times[seed] - self.instance.ready_times[node])
                    demand_diff = abs(self.instance.demands[seed] - self.instance.demands[node])
                    relatedness.append((dist + time_diff * 0.1 + demand_diff * 0.1, node))

            if not relatedness:
                break
            relatedness.sort()
            nodes_to_remove.append(relatedness[0][1])

        for i, route in enumerate(solution.routes):
            solution.routes[i] = [n for n in route if n not in nodes_to_remove]
        solution.routes = [r for r in solution.routes if r]
        return solution, nodes_to_remove

    def greedy_insert(self, partial_solution_with_removed):
        solution, removed_nodes = partial_solution_with_removed
        inst = self.instance

        for node in removed_nodes[:]: # Iterate copy to allow modification
            best_cost = float('inf')
            best_position = None
            best_route_idx = None

            # Try inserting into existing routes
            for route_idx, route in enumerate(solution.routes):
                for pos in range(len(route) + 1):
                    prev = route[pos-1] if pos > 0 else 0
                    next_node = route[pos] if pos < len(route) else 0

                    cost = (inst.distance(prev, node) + inst.distance(node, next_node)
                           - inst.distance(prev, next_node))

                    if cost < best_cost:
                        test_route = route[:pos] + [node] + route[pos:]
                        _, _, feasible = compute_route_metrics(test_route, inst)
                        if feasible:
                            best_cost = cost
                            best_position = pos
                            best_route_idx = route_idx

            # Try creating new route
            new_route_cost = inst.distance(0, node) + inst.distance(node, 0)
            if new_route_cost < best_cost:
                 test_route = [node]
                 _, _, feasible = compute_route_metrics(test_route, inst)
                 if feasible:
                    best_cost = new_route_cost
                    best_position = 0
                    best_route_idx = len(solution.routes)

            if best_route_idx is not None:
                if best_route_idx < len(solution.routes):
                    solution.routes[best_route_idx].insert(best_position, node)
                else:
                    solution.routes.append([node])
            else:
                 # Should not happen if new route is always feasible
                 solution.routes.append([node]) 

        return Solution(solution.routes, inst)

    def regret_insert(self, partial_solution_with_removed):
        solution, removed_nodes = partial_solution_with_removed
        inst = self.instance
        
        # Working copy of removed nodes
        remaining_nodes = list(removed_nodes)

        while remaining_nodes:
            best_regret = -float('inf')
            best_node = None
            best_insert = None

            for node in remaining_nodes:
                costs = []
                # Check existing routes
                for route_idx, route in enumerate(solution.routes):
                    for pos in range(len(route) + 1):
                        prev = route[pos-1] if pos > 0 else 0
                        next_node = route[pos] if pos < len(route) else 0

                        cost = (inst.distance(prev, node) + inst.distance(node, next_node)
                               - inst.distance(prev, next_node))

                        test_route = route[:pos] + [node] + route[pos:]
                        _, _, feasible = compute_route_metrics(test_route, inst)
                        if feasible:
                            costs.append((cost, route_idx, pos))
                
                # Check new route
                new_cost = inst.distance(0, node) + inst.distance(node, 0)
                test_route = [node]
                _, _, feasible = compute_route_metrics(test_route, inst)
                if feasible:
                     costs.append((new_cost, len(solution.routes), 0))

                costs.sort() # Sort by cost ascending

                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                    if regret > best_regret:
                        best_regret = regret
                        best_node = node
                        best_insert = costs[0]
                elif len(costs) == 1:
                    # Only one feasible insertion
                    if best_node is None or costs[0][0] > best_regret: # Simple fallback logic
                        best_node = node
                        best_insert = costs[0]
                        best_regret = float('inf') 

            if best_node is not None and best_insert is not None:
                cost, route_idx, pos = best_insert
                if route_idx < len(solution.routes):
                    solution.routes[route_idx].insert(pos, best_node)
                else:
                    solution.routes.append([best_node])
                remaining_nodes.remove(best_node)
            else:
                # Cannot insert node anywhere feasible (should use new route logic above)
                for node in remaining_nodes:
                    solution.routes.append([node])
                break

        return Solution(solution.routes, inst)
    