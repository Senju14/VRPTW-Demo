"""ALNS solver for VRPTW."""

import random
import math
import time
from copy import deepcopy
from typing import List, Tuple, Optional

from .vrptw_types import VRPTWInstance, Solution


class ALNSSolver:
    """Adaptive Large Neighborhood Search solver for VRPTW."""
    
    def __init__(self, instance: VRPTWInstance, time_limit: float = 30.0):
        self.instance = instance
        self.time_limit = time_limit
    
    def solve(self, initial_routes: Optional[List[List[int]]] = None) -> Solution:
        """Run ALNS optimization."""
        if initial_routes:
            routes = deepcopy(initial_routes)
        else:
            routes = self._greedy_construction()
        
        self._ensure_all_visited(routes)
        
        best_solution = Solution(routes, self.instance)
        best_cost = best_solution.cost
        current_routes = deepcopy(routes)
        current_cost = best_cost
        
        temp = 100.0
        start = time.time()
        
        while time.time() - start < self.time_limit:
            # Try route merge occasionally
            if random.random() < 0.1:
                self._try_merge(current_routes)
                current_routes = [r for r in current_routes if r]
            
            saved = deepcopy(current_routes)
            
            # Destroy: remove random customers
            all_custs = [c for r in current_routes for c in r]
            if len(all_custs) <= 2:
                break
            
            n_remove = random.randint(2, min(8, len(all_custs) // 3 + 1))
            removed = random.sample(all_custs, n_remove)
            
            for r in current_routes:
                for c in removed:
                    if c in r:
                        r.remove(c)
            
            # Repair: reinsert customers at best positions
            for cust in removed:
                best_inc, best_r, best_pos = float('inf'), -1, -1
                
                for i, route in enumerate(current_routes):
                    for j in range(len(route) + 1):
                        inc = self._insertion_cost(route, cust, j)
                        if inc < best_inc:
                            best_inc, best_r, best_pos = inc, i, j
                
                if best_r >= 0 and best_inc < float('inf'):
                    current_routes[best_r].insert(best_pos, cust)
                else:
                    current_routes.append([cust])
            
            new_solution = Solution(current_routes, self.instance)
            
            if not new_solution.feasible:
                current_routes = saved
                continue
            
            # Accept/reject
            if new_solution.cost < best_cost:
                best_solution = new_solution
                best_cost = new_solution.cost
                current_cost = new_solution.cost
            elif new_solution.cost < current_cost:
                current_cost = new_solution.cost
            elif random.random() < math.exp((current_cost - new_solution.cost) / max(temp, 0.01)):
                current_cost = new_solution.cost
            else:
                current_routes = saved
            
            temp *= 0.9995
        
        return best_solution
    
    def _greedy_construction(self) -> List[List[int]]:
        """Build initial solution using nearest neighbor."""
        routes = []
        unvisited = set(range(1, self.instance.n_customers + 1))
        inst = self.instance
        all_nodes = [inst.depot] + inst.customers
        
        while unvisited:
            route = []
            current = 0
            current_time = 0.0
            current_load = 0.0
            
            while True:
                best_cust, best_dist = None, float('inf')
                
                for idx in unvisited:
                    node = all_nodes[idx]
                    d = inst.distance(current, idx)
                    arrival = max(current_time + d, node.ready_time)
                    
                    if (arrival <= node.due_date and 
                        current_load + node.demand <= inst.vehicle_capacity and
                        d < best_dist):
                        best_dist, best_cust = d, idx
                
                if best_cust is None:
                    break
                
                node = all_nodes[best_cust]
                current_time = max(current_time + inst.distance(current, best_cust), node.ready_time)
                current_time += node.service_time
                current_load += node.demand
                current = best_cust
                route.append(best_cust)
                unvisited.remove(best_cust)
            
            if route:
                routes.append(route)
        
        return routes
    
    def _ensure_all_visited(self, routes: List[List[int]]):
        """Make sure all customers are in some route."""
        visited = {c for r in routes for c in r}
        all_custs = set(range(1, self.instance.n_customers + 1))
        
        for c in all_custs - visited:
            inserted = False
            for route in routes:
                for pos in range(len(route) + 1):
                    if self._insertion_cost(route, c, pos) < float('inf'):
                        route.insert(pos, c)
                        inserted = True
                        break
                if inserted:
                    break
            if not inserted:
                routes.append([c])
    
    def _insertion_cost(self, route: List[int], cust: int, pos: int) -> float:
        """Calculate cost increase of inserting customer at position."""
        test_route = route[:pos] + [cust] + route[pos:]
        test_sol = Solution([test_route], self.instance)
        
        if not test_sol.feasible:
            return float('inf')
        
        old_sol = Solution([route], self.instance) if route else Solution([[]], self.instance)
        return test_sol.cost - old_sol.cost
    
    def _try_merge(self, routes: List[List[int]]) -> bool:
        """Try to merge two routes into one."""
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if not routes[i] or not routes[j]:
                    continue
                
                merged = routes[i] + routes[j]
                test_sol = Solution([merged], self.instance)
                
                if test_sol.feasible:
                    routes[i] = merged
                    routes[j] = []
                    return True
        
        return False
