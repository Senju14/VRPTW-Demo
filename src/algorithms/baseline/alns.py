"""ALNS solver for VRPTW (Robust Baseline Implementation)."""

import random
import math
import time
import numpy as np
from copy import deepcopy
from typing import List, Tuple, Optional, Dict
from ...schemas import VRPTWInstance, Solution


class ALNSSolver:
    """
    Adaptive Large Neighborhood Search solver for VRPTW.
    Features: Shaw, Worst, Random Removal; Greedy, Regret-2 Insertion; Adaptive Weighting.
    """
    
    def __init__(self, instance: VRPTWInstance, time_limit: float = 30.0):
        self.instance = instance
        self.time_limit = time_limit
        
        # ALNS Parameters
        self.rho = 0.1  # Reaction factor for weight updates
        self.sigma1 = 33  # Score for new global best
        self.sigma2 = 20  # Score for better than current
        self.sigma3 = 13  # Score for accepted solution
        
        # Operators
        self.destroy_ops = [self._random_removal, self._shaw_removal, self._worst_removal]
        self.repair_ops = [self._greedy_insertion, self._regret_2_insertion]
        
        # Adaptive tracking
        self.d_weights = np.ones(len(self.destroy_ops))
        self.r_weights = np.ones(len(self.repair_ops))
        self.d_scores = np.zeros(len(self.destroy_ops))
        self.r_scores = np.zeros(len(self.repair_ops))
        self.d_counts = np.zeros(len(self.destroy_ops))
        self.r_counts = np.zeros(len(self.repair_ops))
        
    def solve(self, initial_routes: Optional[List[List[int]]] = None) -> Solution:
        """Run ALNS optimization with adaptive weighting."""
        # 1. Initialization
        if initial_routes:
            routes = deepcopy(initial_routes)
        else:
            routes = self._greedy_construction()
        
        self._ensure_all_visited(routes)
        
        best_sol = Solution(routes, self.instance)
        curr_sol = best_sol
        
        # SA Parameters
        temp = self._init_temp(best_sol.cost)
        cooling_rate = 0.9997
        
        start_time = time.time()
        it = 0
        
        while time.time() - start_time < self.time_limit:
            it += 1
            
            # 2. Select operators based on weights
            d_idx = self._select_operator(self.d_weights)
            r_idx = self._select_operator(self.r_weights)
            
            d_op = self.destroy_ops[d_idx]
            r_op = self.repair_ops[r_idx]
            
            # 3. Destroy and Repair
            it_routes = [r[:] for r in curr_sol.routes if r]
            removed = d_op(it_routes)
            r_op(it_routes, removed)
            
            new_sol = Solution(it_routes, self.instance)
            
            # 4. Acceptance Criterion (Simulated Annealing)
            score = 0
            if new_sol.cost < best_sol.cost:
                best_sol = new_sol
                curr_sol = new_sol
                score = self.sigma1
            elif new_sol.cost < curr_sol.cost:
                curr_sol = new_sol
                score = self.sigma2
            else:
                p = math.exp((curr_sol.cost - new_sol.cost) / max(temp, 0.001))
                if random.random() < p:
                    curr_sol = new_sol
                    score = self.sigma3
            
            # 5. Update scores and weights
            self.d_scores[d_idx] += score
            self.r_scores[r_idx] += score
            self.d_counts[d_idx] += 1
            self.r_counts[r_idx] += 1
            
            if it % 100 == 0:
                self._update_weights()
            
            temp *= cooling_rate
            
            # Periodically try to cleanup routes
            if it % 500 == 0:
                curr_sol.routes = [r for r in curr_sol.routes if r]
        
        return best_sol

    # =========================================================================
    # DESTROY OPERATORS
    # =========================================================================

    def _random_removal(self, routes: List[List[int]], n_rem: int = None) -> List[int]:
        """Classic random removal."""
        all_custs = [c for r in routes for c in r]
        if not all_custs: return []
        
        n_rem = n_rem or random.randint(min(4, len(all_custs)), min(12, len(all_custs)//2 + 1))
        removed = random.sample(all_custs, n_rem)
        
        for r in routes:
            for c in removed:
                if c in r: r.remove(c)
        return removed

    def _shaw_removal(self, routes: List[List[int]], n_rem: int = None) -> List[int]:
        """Related removal - removes customers similar to a seed customer."""
        all_custs = [c for r in routes for c in r]
        if not all_custs: return []
        
        n_rem = n_rem or random.randint(min(4, len(all_custs)), min(12, len(all_custs)//2 + 1))
        seed = random.choice(all_custs)
        
        # Calculate relatedness to seed
        relatedness = []
        c1 = self.instance.customers[seed-1]
        for cid in all_custs:
            if cid == seed: continue
            c2 = self.instance.customers[cid-1]
            # Similarity = normalized distance + normalized time start difference
            dist = self.instance.distance(seed, cid) / self.instance.max_coord
            time_diff = abs(c1.ready_time - c2.ready_time) / self.instance.max_time
            relatedness.append((cid, dist + time_diff))
            
        relatedness.sort(key=lambda x: x[1])
        removed = [seed] + [x[0] for x in relatedness[:n_rem-1]]
        
        for r in routes:
            for c in removed:
                if c in r: r.remove(c)
        return removed

    def _worst_removal(self, routes: List[List[int]], n_rem: int = None) -> List[int]:
        """Removes customers with the highest insertion cost in their current routes."""
        cust_costs = []
        for i, route in enumerate(routes):
            for j, cid in enumerate(route):
                # Simple approximation of cost: distance from prev to cid + cid to next
                prev = route[j-1] if j > 0 else 0
                nxt = route[j+1] if j < len(route)-1 else 0
                cost = (self.instance.distance(prev, cid) + 
                        self.instance.distance(cid, nxt) - 
                        self.instance.distance(prev, nxt))
                cust_costs.append((cid, cost))
        
        cust_costs.sort(key=lambda x: x[1], reverse=True)
        
        n_rem = n_rem or random.randint(min(4, len(cust_costs)), min(12, len(cust_costs)//2 + 1))
        removed = [x[0] for x in cust_costs[:n_rem]]
        
        for r in routes:
            for c in removed:
                if c in r: r.remove(c)
        return removed

    # =========================================================================
    # REPAIR OPERATORS
    # =========================================================================

    def _greedy_insertion(self, routes: List[List[int]], removed: List[int]):
        """Standard greedy insertion."""
        random.shuffle(removed)
        for cid in removed:
            best_r, best_pos, best_cost = self._find_best_insertion(routes, cid)
            if best_r != -1:
                routes[best_r].insert(best_pos, cid)
            else:
                routes.append([cid])

    def _regret_2_insertion(self, routes: List[List[int]], removed: List[int]):
        """Regret-2 insertion - prioritizes customers with largest gap between 1st and 2nd best insertion."""
        while removed:
            regret_list = []
            for cid in removed:
                costs = []
                for r_idx, route in enumerate(routes):
                    for pos in range(len(route) + 1):
                        cost = self._calc_insertion_increase(route, cid, pos)
                        if cost < float('inf'):
                            costs.append((r_idx, pos, cost))
                
                # Add a potential new route option
                costs.append((len(routes), 0, self._calc_insertion_increase([], cid, 0)))
                
                costs.sort(key=lambda x: x[2])
                regret = costs[1][2] - costs[0][2] if len(costs) > 1 else costs[0][2]
                regret_list.append((cid, regret, costs[0]))
            
            # Select customer with max regret
            regret_list.sort(key=lambda x: x[1], reverse=True)
            best_cid, _, (r_idx, pos, _) = regret_list[0]
            
            if r_idx == len(routes):
                routes.append([best_cid])
            else:
                routes[r_idx].insert(pos, best_cid)
            
            removed.remove(best_cid)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _find_best_insertion(self, routes: List[List[int]], cid: int) -> Tuple[int, int, float]:
        best_r, best_pos, best_cost = -1, -1, float('inf')
        for i, route in enumerate(routes):
            for j in range(len(route) + 1):
                inc = self._calc_insertion_increase(route, cid, j)
                if inc < best_cost:
                    best_cost, best_r, best_pos = inc, i, j
        return best_r, best_pos, best_cost

    def _calc_insertion_increase(self, route: List[int], cid: int, pos: int) -> float:
        """Lightweight feasibility check and cost calculation."""
        inst = self.instance
        cust = inst.customers[cid-1]
        
        # 1. Quick capacity check
        load = sum(inst.customers[idx-1].demand for idx in route) + cust.demand
        if load > inst.vehicle_capacity:
            return float('inf')
        
        # 2. Sequential time window check
        test_route = route[:pos] + [cid] + route[pos:]
        time = 0.0
        prev = 0
        all_nodes = [inst.depot] + inst.customers
        for nid in test_route:
            node = all_nodes[nid]
            time += inst.distance(prev, nid)
            time = max(time, node.ready_time)
            if time > node.due_date:
                return float('inf')
            time += node.service_time
            prev = nid
        
        # 3. Calculate cost increase
        prev_node = route[pos-1] if pos > 0 else 0
        next_node = route[pos] if pos < len(route) else 0
        inc = (inst.distance(prev_node, cid) + 
               inst.distance(cid, next_node) - 
               inst.distance(prev_node, next_node))
        return float(inc)

    def _greedy_construction(self) -> List[List[int]]:
        """Nearest Neighbor with TW feasibility."""
        routes = []
        unvisited = set(range(1, self.instance.n_customers + 1))
        while unvisited:
            route = []
            curr, curr_t, curr_l = 0, 0.0, 0.0
            while True:
                best_c, best_d = None, float('inf')
                for cid in unvisited:
                    c = self.instance.customers[cid-1]
                    d = self.instance.distance(curr, cid)
                    arr = max(curr_t + d, c.ready_time)
                    if arr <= c.due_date and curr_l + c.demand <= self.instance.vehicle_capacity:
                        if d < best_d: best_d, best_c = d, cid
                
                if best_c is None: break
                
                c = self.instance.customers[best_c-1]
                curr_t = max(curr_t + self.instance.distance(curr, best_c), c.ready_time) + c.service_time
                curr_l += c.demand
                curr = best_c
                route.append(best_c)
                unvisited.remove(best_c)
            if route: routes.append(route)
        return routes

    def _ensure_all_visited(self, routes: List[List[int]]):
        visited = {c for r in routes for c in r}
        for cid in set(range(1, self.instance.n_customers + 1)) - visited:
            best_r, best_pos, _ = self._find_best_insertion(routes, cid)
            if best_r != -1: routes[best_r].insert(best_pos, cid)
            else: routes.append([cid])

    def _select_operator(self, weights: np.ndarray) -> int:
        p = weights / weights.sum()
        return np.random.choice(len(weights), p=p)

    def _update_weights(self):
        """Update adaptive weights based on scores and counts."""
        for i in range(len(self.d_weights)):
            if self.d_counts[i] > 0:
                score = self.d_scores[i] / self.d_counts[i]
                self.d_weights[i] = (1 - self.rho) * self.d_weights[i] + self.rho * score
        
        for i in range(len(self.r_weights)):
            if self.r_counts[i] > 0:
                score = self.r_scores[i] / self.r_counts[i]
                self.r_weights[i] = (1 - self.rho) * self.r_weights[i] + self.rho * score
        
        # Reset scores and counts for next period
        self.d_scores.fill(0); self.r_scores.fill(0)
        self.d_counts.fill(0); self.r_counts.fill(0)

    def _init_temp(self, init_cost: float) -> float:
        """Initial temperature to allow ~50% acceptance of 5% worse solutions."""
        return -0.05 * init_cost / math.log(0.5)
