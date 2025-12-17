import random

class ALNS:
    def __init__(self, instance, max_iterations=100):
        self.instance = instance
        self.max_iterations = max_iterations
        self.destroy_size = int(0.3 * instance.n)
        self.temp = 100
        self.temp_decay = 0.99
    
    def solve(self):
        from .utils import Solution
        
        current = self._create_initial_solution()
        best = current.copy()
        
        for _ in range(self.max_iterations):
            destroyed = self._random_removal(current.copy())
            candidate = self._greedy_insert(destroyed)
            
            if candidate.feasible:
                if candidate.cost < current.cost:
                    current = candidate
                    if candidate.cost < best.cost:
                        best = candidate.copy()
            
            self.temp *= self.temp_decay
        
        return best
    
    def _create_initial_solution(self):
        from .utils import Solution
        
        inst = self.instance
        unvisited = list(range(1, inst.n + 1))
        routes = []
        
        while unvisited:
            route = []
            load = 0
            time = 0
            current = 0
            
            while unvisited:
                feasible = [n for n in unvisited 
                           if load + inst.demands[n] <= inst.capacity
                           and time + inst.distance(current, n) <= inst.due_times[n]]
                
                if not feasible:
                    break
                
                next_node = min(feasible, key=lambda n: inst.distance(current, n))
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
    
    def _random_removal(self, solution):
        nodes = [n for r in solution.routes for n in r]
        if not nodes:
            return solution, []
        
        to_remove = random.sample(nodes, min(self.destroy_size, len(nodes)))
        for route in solution.routes:
            solution.routes[solution.routes.index(route)] = [n for n in route if n not in to_remove]
        solution.routes = [r for r in solution.routes if r]
        return solution, to_remove
    
    def _greedy_insert(self, partial):
        from .utils import Solution
        
        solution, removed = partial
        inst = self.instance
        
        for node in removed:
            best_cost = float('inf')
            best_pos = None
            best_route = None
            
            for ridx, route in enumerate(solution.routes):
                for pos in range(len(route) + 1):
                    prev = route[pos-1] if pos > 0 else 0
                    next_node = route[pos] if pos < len(route) else 0
                    cost = inst.distance(prev, node) + inst.distance(node, next_node) - inst.distance(prev, next_node)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos
                        best_route = ridx
            
            if best_route is not None:
                solution.routes[best_route].insert(best_pos, node)
            else:
                solution.routes.append([node])
        
        return Solution(solution.routes, inst)
    