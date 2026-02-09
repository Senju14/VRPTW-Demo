"""
ALNS Baseline for VRPTW - Enhanced Version
Based on Ropke & Pisinger (2006)

Enhancements:
- 2-opt local search
- Iteration-based stopping
- Noise in insertion cost
- Parallel multi-start
"""

import os
import random
import math
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from pathlib import Path

# =============================================================================
# HYPERPARAMETERS (from Ropke & Pisinger 2006)
# =============================================================================

# Adaptive Weight Update Parameters
SIGMA1 = 33     # New global best solution found
SIGMA2 = 20     # Better than current solution  
SIGMA3 = 13     # Solution accepted (worse but accepted by SA)
RHO = 0.1       # Reaction factor for weight updates
ETA_S = 100     # Weight update period (iterations)

# Simulated Annealing Parameters
ALPHA = 0.99975  # Cooling rate (adjusted for more iterations)
W_START = 0.05   # Initial temperature parameter

# Removal Parameters
Q_MIN = 4       # Minimum customers to remove
Q_MAX = 100     # Maximum (will be capped at 40% of customers)

# Shaw Removal Parameters (Relatedness)
PHI = 9         # Distance weight
CHI = 3         # Time weight  
PSI = 2         # Demand weight
P_SHAW = 6      # Randomness parameter (higher = more deterministic)

# Worst Removal Parameters
P_WORST = 3     # Randomness parameter

# Noise Parameter (from Ropke & Pisinger)
NOISE_PARAM = 0.025  # Max noise level for insertion cost
USE_NOISE = True     # Enable/disable noise

# Iteration-based stopping
MAX_ITERATIONS = 25000
MAX_NO_IMPROVE = 5000  # Restart/stop if no improvement

# Parallel settings
N_PARALLEL_RUNS = 4  # Number of parallel ALNS runs


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Customer:
    """A customer node with location, demand, and time windows."""
    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float


@dataclass
class VRPTWInstance:
    """A complete VRPTW problem instance."""
    name: str
    depot: Customer
    customers: List[Customer]
    vehicle_capacity: float
    num_vehicles: int
    dist_matrix: np.ndarray = field(default=None, repr=False)
    
    def __post_init__(self):
        self._build_distance_matrix()
        all_nodes = [self.depot] + self.customers
        self.max_coord = max(max(c.x for c in all_nodes), max(c.y for c in all_nodes))
        self.max_time = max(c.due_date for c in all_nodes)
        self.max_demand = max((c.demand for c in self.customers), default=1.0)
        # Max distance for noise calculation
        self.max_distance = np.max(self.dist_matrix)
    
    def _build_distance_matrix(self):
        """Precompute Euclidean distance matrix for all nodes."""
        n = len(self.customers) + 1
        self.dist_matrix = np.zeros((n, n), dtype=np.float64)
        all_nodes = [self.depot] + self.customers
        
        for i, n1 in enumerate(all_nodes):
            for j, n2 in enumerate(all_nodes):
                self.dist_matrix[i, j] = math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
    
    def distance(self, i: int, j: int) -> float:
        """Get distance between nodes i and j (0 = depot)."""
        return float(self.dist_matrix[i, j])
    
    def get_customer(self, cid: int) -> Customer:
        """Get customer by ID (1-indexed)."""
        return self.customers[cid - 1]
    
    @property
    def n_customers(self) -> int:
        return len(self.customers)


@dataclass
class Solution:
    """A solution to a VRPTW instance."""
    routes: List[List[int]]
    instance: VRPTWInstance
    cost: float = field(default=0.0, init=False)
    feasible: bool = field(default=True, init=False)
    
    def __post_init__(self):
        self.cost = self._compute_cost()
        self.feasible = self._check_feasibility()
    
    def _compute_cost(self) -> float:
        """Total distance of all routes."""
        total = 0.0
        for route in self.routes:
            if not route:
                continue
            total += self.instance.distance(0, route[0])
            for i in range(len(route) - 1):
                total += self.instance.distance(route[i], route[i+1])
            total += self.instance.distance(route[-1], 0)
        return total
    
    def _check_feasibility(self) -> bool:
        """Check capacity and time window constraints."""
        inst = self.instance
        
        for route in self.routes:
            if not route:
                continue
            
            load = sum(inst.get_customer(cid).demand for cid in route)
            if load > inst.vehicle_capacity:
                return False
            
            time = 0.0
            prev = 0
            for cid in route:
                cust = inst.get_customer(cid)
                time += inst.distance(prev, cid)
                time = max(time, cust.ready_time)
                if time > cust.due_date:
                    return False
                time += cust.service_time
                prev = cid
        
        return True
    
    @property
    def num_vehicles(self) -> int:
        return len([r for r in self.routes if r])
    
    def copy(self) -> 'Solution':
        return Solution([r[:] for r in self.routes], self.instance)
    
    def recalculate(self):
        self.cost = self._compute_cost()
        self.feasible = self._check_feasibility()


# =============================================================================
# DATA LOADING
# =============================================================================

def parse_solomon_file(filepath: str) -> VRPTWInstance:
    """Parse Solomon format benchmark file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    name = lines[0].strip()
    vehicle_line = lines[4].split()
    num_vehicles = int(vehicle_line[0])
    capacity = float(vehicle_line[1])
    
    nodes = []
    for line in lines[9:]:
        parts = line.split()
        if len(parts) >= 7:
            nodes.append({
                'id': int(parts[0]),
                'x': float(parts[1]),
                'y': float(parts[2]),
                'demand': float(parts[3]),
                'ready_time': float(parts[4]),
                'due_date': float(parts[5]),
                'service_time': float(parts[6])
            })
    
    depot_data = nodes[0]
    depot = Customer(
        id=0, x=depot_data['x'], y=depot_data['y'],
        demand=0, ready_time=depot_data['ready_time'],
        due_date=depot_data['due_date'], service_time=0
    )
    
    customers = [
        Customer(id=n['id'], x=n['x'], y=n['y'], demand=n['demand'],
                 ready_time=n['ready_time'], due_date=n['due_date'],
                 service_time=n['service_time'])
        for n in nodes[1:]
    ]
    
    return VRPTWInstance(
        name=name,
        depot=depot,
        customers=customers,
        vehicle_capacity=capacity,
        num_vehicles=num_vehicles
    )


def get_solomon_data_path() -> str:
    """Get path to Solomon benchmark data."""
    current_dir = Path(__file__).parent
    data_path = current_dir.parent / "data" / "Solomon"
    
    if data_path.exists():
        return str(data_path)
    
    abs_path = Path(r"d:\GitHub\VRPTW-Demo\data\Solomon")
    if abs_path.exists():
        return str(abs_path)
    
    raise FileNotFoundError("Solomon benchmark data not found!")


def load_rc_instances() -> Dict[str, VRPTWInstance]:
    """Load all 16 RC Solomon instances."""
    data_path = get_solomon_data_path()
    instances = {}
    
    rc_files = [
        "rc101.txt", "rc102.txt", "rc103.txt", "rc104.txt",
        "rc105.txt", "rc106.txt", "rc107.txt", "rc108.txt",
        "rc201.txt", "rc202.txt", "rc203.txt", "rc204.txt",
        "rc205.txt", "rc206.txt", "rc207.txt", "rc208.txt",
    ]
    
    for fname in rc_files:
        filepath = os.path.join(data_path, fname)
        if os.path.exists(filepath):
            instance = parse_solomon_file(filepath)
            instances[instance.name] = instance
    
    return instances


# =============================================================================
# INITIAL SOLUTION CONSTRUCTION
# =============================================================================

def nearest_neighbor_construction(instance: VRPTWInstance) -> Solution:
    """Construct initial solution using Nearest Neighbor heuristic."""
    routes = []
    unvisited = set(range(1, instance.n_customers + 1))
    
    while unvisited:
        route = []
        curr_node = 0
        curr_time = 0.0
        curr_load = 0.0
        
        while True:
            best_cust = None
            best_dist = float('inf')
            
            for cid in unvisited:
                cust = instance.get_customer(cid)
                dist = instance.distance(curr_node, cid)
                arrival = curr_time + dist
                
                if arrival > cust.due_date:
                    continue
                if curr_load + cust.demand > instance.vehicle_capacity:
                    continue
                
                service_end = max(arrival, cust.ready_time) + cust.service_time
                if service_end + instance.distance(cid, 0) > instance.depot.due_date:
                    continue
                
                if dist < best_dist:
                    best_dist = dist
                    best_cust = cid
            
            if best_cust is None:
                break
            
            cust = instance.get_customer(best_cust)
            curr_time = max(curr_time + best_dist, cust.ready_time) + cust.service_time
            curr_load += cust.demand
            curr_node = best_cust
            route.append(best_cust)
            unvisited.remove(best_cust)
        
        if route:
            routes.append(route)
    
    return Solution(routes, instance)


# =============================================================================
# LOCAL SEARCH OPERATORS
# =============================================================================

def check_route_feasibility(route: List[int], instance: VRPTWInstance) -> bool:
    """Check if a route is feasible (time windows and capacity)."""
    if not route:
        return True
    
    load = sum(instance.get_customer(cid).demand for cid in route)
    if load > instance.vehicle_capacity:
        return False
    
    time = 0.0
    prev = 0
    for cid in route:
        cust = instance.get_customer(cid)
        time += instance.distance(prev, cid)
        time = max(time, cust.ready_time)
        if time > cust.due_date:
            return False
        time += cust.service_time
        prev = cid
    
    if time + instance.distance(route[-1], 0) > instance.depot.due_date:
        return False
    
    return True


def route_cost(route: List[int], instance: VRPTWInstance) -> float:
    """Calculate cost of a single route."""
    if not route:
        return 0.0
    cost = instance.distance(0, route[0])
    for i in range(len(route) - 1):
        cost += instance.distance(route[i], route[i+1])
    cost += instance.distance(route[-1], 0)
    return cost


def two_opt_route(route: List[int], instance: VRPTWInstance) -> List[int]:
    """
    2-opt local search within a single route.
    Reverses segments to reduce distance while maintaining feasibility.
    """
    if len(route) < 4:
        return route
    
    improved = True
    best_route = route[:]
    best_cost = route_cost(best_route, instance)
    
    while improved:
        improved = False
        for i in range(len(best_route) - 2):
            for j in range(i + 2, len(best_route)):
                # Create new route by reversing segment [i+1:j+1]
                new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                
                if check_route_feasibility(new_route, instance):
                    new_cost = route_cost(new_route, instance)
                    if new_cost < best_cost - 1e-6:
                        best_route = new_route
                        best_cost = new_cost
                        improved = True
                        break
            if improved:
                break
    
    return best_route


def local_search_2opt(solution: Solution) -> Solution:
    """Apply 2-opt to all routes in the solution."""
    sol = solution.copy()
    
    for i in range(len(sol.routes)):
        if len(sol.routes[i]) >= 4:
            sol.routes[i] = two_opt_route(sol.routes[i], sol.instance)
    
    sol.recalculate()
    return sol


def or_opt_route(route: List[int], instance: VRPTWInstance) -> List[int]:
    """
    Or-opt: relocate 1-3 consecutive customers to better positions.
    """
    if len(route) < 3:
        return route
    
    improved = True
    best_route = route[:]
    best_cost = route_cost(best_route, instance)
    
    while improved:
        improved = False
        # Try relocating segments of size 1, 2, 3
        for seg_size in [1, 2, 3]:
            if seg_size > len(best_route):
                continue
            for i in range(len(best_route) - seg_size + 1):
                segment = best_route[i:i+seg_size]
                remaining = best_route[:i] + best_route[i+seg_size:]
                
                for j in range(len(remaining) + 1):
                    new_route = remaining[:j] + segment + remaining[j:]
                    
                    if check_route_feasibility(new_route, instance):
                        new_cost = route_cost(new_route, instance)
                        if new_cost < best_cost - 1e-6:
                            best_route = new_route
                            best_cost = new_cost
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    return best_route


def local_search_full(solution: Solution) -> Solution:
    """Apply 2-opt and Or-opt to all routes."""
    sol = solution.copy()
    
    for i in range(len(sol.routes)):
        if len(sol.routes[i]) >= 3:
            sol.routes[i] = or_opt_route(sol.routes[i], sol.instance)
        if len(sol.routes[i]) >= 4:
            sol.routes[i] = two_opt_route(sol.routes[i], sol.instance)
    
    sol.recalculate()
    return sol


# =============================================================================
# DESTROY OPERATORS
# =============================================================================

def random_removal(solution: Solution, q: int) -> Tuple[Solution, List[int]]:
    """Random Removal: Remove q random customers."""
    sol = solution.copy()
    all_custs = [c for r in sol.routes for c in r]
    
    if len(all_custs) < q:
        q = len(all_custs)
    
    removed = random.sample(all_custs, q)
    
    for route in sol.routes:
        for c in removed:
            if c in route:
                route.remove(c)
    
    sol.routes = [r for r in sol.routes if r]
    sol.recalculate()
    
    return sol, removed


def shaw_removal(solution: Solution, q: int) -> Tuple[Solution, List[int]]:
    """
    Shaw Removal (Algorithm 2 from Ropke & Pisinger 2006).
    Remove related requests based on distance, time, and demand similarity.
    """
    sol = solution.copy()
    inst = sol.instance
    all_custs = [c for r in sol.routes for c in r]
    
    if not all_custs:
        return sol, []
    
    q = min(q, len(all_custs))
    
    seed = random.choice(all_custs)
    D = [seed]
    
    while len(D) < q:
        r = random.choice(D)
        r_cust = inst.get_customer(r)
        
        L = [c for c in all_custs if c not in D]
        if not L:
            break
        
        relatedness = []
        for cid in L:
            cust = inst.get_customer(cid)
            dist = inst.distance(r, cid) / inst.max_coord
            time_diff = abs(r_cust.ready_time - cust.ready_time) / inst.max_time
            demand_diff = abs(r_cust.demand - cust.demand) / max(inst.max_demand, 1.0)
            R = PHI * dist + CHI * time_diff + PSI * demand_diff
            relatedness.append((cid, R))
        
        relatedness.sort(key=lambda x: x[1])
        
        y = random.random()
        idx = int(y ** P_SHAW * len(relatedness))
        idx = min(idx, len(relatedness) - 1)
        
        D.append(relatedness[idx][0])
    
    for route in sol.routes:
        for c in D:
            if c in route:
                route.remove(c)
    
    sol.routes = [r for r in sol.routes if r]
    sol.recalculate()
    
    return sol, D


def worst_removal(solution: Solution, q: int) -> Tuple[Solution, List[int]]:
    """
    Worst Removal (Algorithm 3 from Ropke & Pisinger 2006).
    Remove one request at a time, recalculating cost after each removal.
    """
    sol = solution.copy()
    inst = sol.instance
    removed = []
    
    while len(removed) < q:
        costs = []
        for route_idx, route in enumerate(sol.routes):
            for pos, cid in enumerate(route):
                prev_node = route[pos-1] if pos > 0 else 0
                next_node = route[pos+1] if pos < len(route) - 1 else 0
                
                cost = (inst.distance(prev_node, cid) + 
                       inst.distance(cid, next_node) - 
                       inst.distance(prev_node, next_node))
                costs.append((cid, cost, route_idx))
        
        if not costs:
            break
        
        costs.sort(key=lambda x: x[1], reverse=True)
        
        y = random.random()
        idx = int(y ** P_WORST * len(costs))
        idx = min(idx, len(costs) - 1)
        
        cid, _, route_idx = costs[idx]
        
        sol.routes[route_idx].remove(cid)
        removed.append(cid)
        
        sol.routes = [r for r in sol.routes if r]
    
    sol.recalculate()
    return sol, removed


# =============================================================================
# REPAIR OPERATORS
# =============================================================================

def calc_insertion_cost(route: List[int], cid: int, pos: int, 
                        instance: VRPTWInstance, use_noise: bool = False) -> float:
    """
    Calculate insertion cost with optional noise.
    Returns float('inf') if infeasible.
    """
    cust = instance.get_customer(cid)
    
    # Capacity check
    load = sum(instance.get_customer(c).demand for c in route) + cust.demand
    if load > instance.vehicle_capacity:
        return float('inf')
    
    # Time window feasibility check
    test_route = route[:pos] + [cid] + route[pos:]
    time = 0.0
    prev = 0
    
    for node_id in test_route:
        node = instance.get_customer(node_id)
        time += instance.distance(prev, node_id)
        time = max(time, node.ready_time)
        if time > node.due_date:
            return float('inf')
        time += node.service_time
        prev = node_id
    
    if time + instance.distance(prev, 0) > instance.depot.due_date:
        return float('inf')
    
    # Calculate cost increase
    prev_node = route[pos-1] if pos > 0 else 0
    next_node = route[pos] if pos < len(route) else 0
    
    cost_increase = (instance.distance(prev_node, cid) + 
                    instance.distance(cid, next_node) - 
                    instance.distance(prev_node, next_node))
    
    # Add noise if enabled
    if use_noise and USE_NOISE:
        noise = NOISE_PARAM * instance.max_distance * random.uniform(-1, 1)
        cost_increase = max(0, cost_increase + noise)
    
    return cost_increase


def find_best_insertion(routes: List[List[int]], cid: int, 
                        instance: VRPTWInstance, use_noise: bool = False) -> Tuple[int, int, float]:
    """Find the best insertion position for customer cid."""
    best_route = -1
    best_pos = -1
    best_cost = float('inf')
    
    for r_idx, route in enumerate(routes):
        for pos in range(len(route) + 1):
            cost = calc_insertion_cost(route, cid, pos, instance, use_noise)
            if cost < best_cost:
                best_cost = cost
                best_route = r_idx
                best_pos = pos
    
    return best_route, best_pos, best_cost


def greedy_insertion(solution: Solution, removed: List[int], use_noise: bool = False) -> Solution:
    """Greedy Insertion with optional noise."""
    sol = solution.copy()
    inst = sol.instance
    
    random.shuffle(removed)
    
    for cid in removed:
        r_idx, pos, cost = find_best_insertion(sol.routes, cid, inst, use_noise)
        
        if r_idx != -1:
            sol.routes[r_idx].insert(pos, cid)
        else:
            sol.routes.append([cid])
    
    sol.recalculate()
    return sol


def regret_k_insertion(solution: Solution, removed: List[int], k: int = 2, use_noise: bool = False) -> Solution:
    """
    Regret-k Insertion: Prioritize customers with high regret value.
    Regret = sum of (k-th best cost - best cost)
    """
    sol = solution.copy()
    inst = sol.instance
    remaining = removed[:]
    
    while remaining:
        regret_values = []
        
        for cid in remaining:
            all_costs = []
            
            for r_idx, route in enumerate(sol.routes):
                for pos in range(len(route) + 1):
                    cost = calc_insertion_cost(route, cid, pos, inst, use_noise)
                    if cost < float('inf'):
                        all_costs.append((r_idx, pos, cost))
            
            new_route_cost = calc_insertion_cost([], cid, 0, inst, use_noise)
            if new_route_cost < float('inf'):
                all_costs.append((len(sol.routes), 0, new_route_cost))
            
            if not all_costs:
                all_costs.append((len(sol.routes), 0, 0))
            
            all_costs.sort(key=lambda x: x[2])
            
            best = all_costs[0]
            regret = 0
            for i in range(1, min(k, len(all_costs))):
                regret += all_costs[i][2] - all_costs[0][2]
            
            regret_values.append((cid, regret, best))
        
        regret_values.sort(key=lambda x: x[1], reverse=True)
        best_cid, _, (r_idx, pos, _) = regret_values[0]
        
        if r_idx >= len(sol.routes):
            sol.routes.append([best_cid])
        else:
            sol.routes[r_idx].insert(pos, best_cid)
        
        remaining.remove(best_cid)
    
    sol.recalculate()
    return sol


def regret_2_insertion(solution: Solution, removed: List[int]) -> Solution:
    """Regret-2 Insertion wrapper."""
    return regret_k_insertion(solution, removed, k=2, use_noise=USE_NOISE)


def regret_3_insertion(solution: Solution, removed: List[int]) -> Solution:
    """Regret-3 Insertion."""
    return regret_k_insertion(solution, removed, k=3, use_noise=USE_NOISE)


# =============================================================================
# ADAPTIVE WEIGHT MECHANISM
# =============================================================================

class AdaptiveWeights:
    """Manages adaptive weights for operator selection."""
    
    def __init__(self, n_destroy: int, n_repair: int):
        self.w_destroy = np.ones(n_destroy)
        self.w_repair = np.ones(n_repair)
        
        self.scores_d = np.zeros(n_destroy)
        self.scores_r = np.zeros(n_repair)
        self.counts_d = np.zeros(n_destroy)
        self.counts_r = np.zeros(n_repair)
    
    def select_operator(self, weights: np.ndarray) -> int:
        """Roulette wheel selection based on weights."""
        probs = weights / weights.sum()
        return np.random.choice(len(weights), p=probs)
    
    def update_score(self, d_idx: int, r_idx: int, score: float):
        """Update scores for selected operators."""
        self.scores_d[d_idx] += score
        self.scores_r[r_idx] += score
        self.counts_d[d_idx] += 1
        self.counts_r[r_idx] += 1
    
    def update_weights(self):
        """Update weights based on accumulated scores."""
        for i in range(len(self.w_destroy)):
            if self.counts_d[i] > 0:
                avg_score = self.scores_d[i] / self.counts_d[i]
                self.w_destroy[i] = (1 - RHO) * self.w_destroy[i] + RHO * avg_score
        
        for i in range(len(self.w_repair)):
            if self.counts_r[i] > 0:
                avg_score = self.scores_r[i] / self.counts_r[i]
                self.w_repair[i] = (1 - RHO) * self.w_repair[i] + RHO * avg_score
        
        self.scores_d.fill(0)
        self.scores_r.fill(0)
        self.counts_d.fill(0)
        self.counts_r.fill(0)


# =============================================================================
# SIMULATED ANNEALING ACCEPTANCE
# =============================================================================

def init_temperature(init_cost: float, w: float = W_START) -> float:
    """Calculate initial temperature for SA."""
    return -w * init_cost / math.log(0.5)


def accept_solution(cost_new: float, cost_current: float, temperature: float) -> bool:
    """Simulated Annealing acceptance criterion."""
    if cost_new < cost_current:
        return True
    
    if temperature <= 0:
        return False
    
    delta = cost_new - cost_current
    probability = math.exp(-delta / temperature)
    return random.random() < probability


# =============================================================================
# MAIN ALNS ALGORITHM
# =============================================================================

def alns_solve(instance: VRPTWInstance, 
               max_iterations: int = MAX_ITERATIONS,
               use_local_search: bool = True,
               verbose: bool = True,
               seed: int = None) -> Tuple[Solution, Dict]:
    """
    Enhanced ALNS Algorithm with:
    - Iteration-based stopping
    - Local search (2-opt, Or-opt)
    - Noise in insertion cost
    
    Args:
        instance: VRPTW problem instance
        max_iterations: Maximum number of iterations
        use_local_search: Apply local search after repair
        verbose: Print progress information
        seed: Random seed for reproducibility
    
    Returns:
        best_solution: Best solution found
        stats: Dictionary with algorithm statistics
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Define operators
    destroy_ops = [random_removal, shaw_removal, worst_removal]
    repair_ops = [greedy_insertion, regret_2_insertion, regret_3_insertion]
    destroy_names = ["Random", "Shaw", "Worst"]
    repair_names = ["Greedy", "Regret-2", "Regret-3"]
    
    # Initialize
    S = nearest_neighbor_construction(instance)
    if use_local_search:
        S = local_search_2opt(S)
    
    S_best = S.copy()
    S_current = S.copy()
    
    # Initialize temperature
    T = init_temperature(S.cost)
    
    # Initialize adaptive weights
    weights = AdaptiveWeights(len(destroy_ops), len(repair_ops))
    
    # Statistics
    stats = {
        'iterations': 0,
        'improvements': 0,
        'accepts': 0,
        'init_cost': S.cost,
        'init_vehicles': S.num_vehicles
    }
    
    start_time = time.time()
    no_improve_count = 0
    last_improve_iter = 0
    
    for iteration in range(1, max_iterations + 1):
        # Check for early termination
        if no_improve_count >= MAX_NO_IMPROVE:
            if verbose:
                print(f"  Early stop: no improvement for {MAX_NO_IMPROVE} iterations")
            break
        
        # 1. Select operators using roulette wheel
        d_idx = weights.select_operator(weights.w_destroy)
        r_idx = weights.select_operator(weights.w_repair)
        
        destroy_op = destroy_ops[d_idx]
        repair_op = repair_ops[r_idx]
        
        # 2. Determine number of customers to remove (4-40% of customers)
        n_custs = sum(len(r) for r in S_current.routes)
        q_max = max(Q_MIN, int(0.4 * n_custs))
        q = random.randint(Q_MIN, min(Q_MAX, q_max))
        
        # 3. Destroy and Repair
        S_destroyed, removed = destroy_op(S_current, q)
        S_new = repair_op(S_destroyed, removed)
        
        # 4. Apply local search only when new global best found
        if use_local_search and S_new.cost < S_best.cost:
            S_new = local_search_2opt(S_new)
        
        # 5. Evaluate and Accept/Reject
        score = 0
        
        if S_new.cost < S_best.cost:
            # New global best
            S_best = S_new.copy()
            S_current = S_new
            score = SIGMA1
            stats['improvements'] += 1
            no_improve_count = 0
            last_improve_iter = iteration
            
            if verbose and (iteration <= 100 or iteration % 100 == 0 or stats['improvements'] <= 50):
                elapsed = time.time() - start_time
                print(f"  Iter {iteration}: New best = {S_best.cost:.2f} "
                      f"({S_best.num_vehicles} vehicles) @ {elapsed:.1f}s")
        
        elif S_new.cost < S_current.cost:
            S_current = S_new
            score = SIGMA2
            stats['accepts'] += 1
            no_improve_count = 0
        
        elif accept_solution(S_new.cost, S_current.cost, T):
            S_current = S_new
            score = SIGMA3
            stats['accepts'] += 1
            no_improve_count += 1
        else:
            no_improve_count += 1
        
        # 6. Update adaptive weights
        weights.update_score(d_idx, r_idx, score)
        
        if iteration % ETA_S == 0:
            weights.update_weights()
        
        # 7. Cool temperature
        T *= ALPHA
    
    # Final local search on best solution
    if use_local_search:
        S_best = local_search_full(S_best)
    
    stats['iterations'] = iteration
    stats['final_cost'] = S_best.cost
    stats['final_vehicles'] = S_best.num_vehicles
    stats['solve_time'] = time.time() - start_time
    stats['destroy_weights'] = dict(zip(destroy_names, weights.w_destroy.tolist()))
    stats['repair_weights'] = dict(zip(repair_names, weights.w_repair.tolist()))
    
    return S_best, stats


def single_run_wrapper(args):
    """Wrapper for parallel execution."""
    instance, max_iterations, use_local_search, seed = args
    solution, stats = alns_solve(
        instance, 
        max_iterations=max_iterations,
        use_local_search=use_local_search,
        verbose=False,
        seed=seed
    )
    return solution, stats


def parallel_alns(instance: VRPTWInstance,
                  n_runs: int = N_PARALLEL_RUNS,
                  max_iterations: int = MAX_ITERATIONS,
                  use_local_search: bool = True,
                  verbose: bool = True) -> Tuple[Solution, Dict]:
    """
    Run multiple ALNS instances in parallel and return the best result.
    
    Args:
        instance: VRPTW problem instance
        n_runs: Number of parallel runs
        max_iterations: Max iterations per run
        use_local_search: Apply local search
        verbose: Print progress
    
    Returns:
        best_solution: Best solution across all runs
        stats: Combined statistics
    """
    if verbose:
        print(f"  Running {n_runs} parallel ALNS instances...")
    
    start_time = time.time()
    
    # Prepare arguments for each run
    args_list = [
        (instance, max_iterations, use_local_search, random.randint(0, 10000) + i)
        for i in range(n_runs)
    ]
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=n_runs) as executor:
        results = list(executor.map(single_run_wrapper, args_list))
    
    # Find best result
    best_solution = None
    best_cost = float('inf')
    total_iterations = 0
    
    for solution, stats in results:
        total_iterations += stats['iterations']
        if solution.cost < best_cost:
            best_cost = solution.cost
            best_solution = solution
            best_stats = stats
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"  Parallel complete: Best = {best_solution.cost:.2f} "
              f"({best_solution.num_vehicles} veh) in {elapsed:.1f}s")
    
    best_stats['total_time'] = time.time() - start_time
    best_stats['total_iterations'] = total_iterations
    
    return best_solution, best_stats


# =============================================================================
# BENCHMARK
# =============================================================================

BKS = {
    'RC101': {'vehicles': 14, 'distance': 1619.8},
    'RC102': {'vehicles': 12, 'distance': 1457.4},
    'RC103': {'vehicles': 11, 'distance': 1258.0},
    'RC104': {'vehicles': 10, 'distance': 1132.3},
    'RC105': {'vehicles': 13, 'distance': 1513.7},
    'RC106': {'vehicles': 11, 'distance': 1372.7},
    'RC107': {'vehicles': 11, 'distance': 1207.8},
    'RC108': {'vehicles': 10, 'distance': 1114.2},
    'RC201': {'vehicles': 4, 'distance': 1261.8},
    'RC202': {'vehicles': 3, 'distance': 1092.3},
    'RC203': {'vehicles': 3, 'distance': 923.7},
    'RC204': {'vehicles': 3, 'distance': 783.5},
    'RC205': {'vehicles': 4, 'distance': 1154.0},
    'RC206': {'vehicles': 3, 'distance': 1051.1},
    'RC207': {'vehicles': 3, 'distance': 962.9},
    'RC208': {'vehicles': 3, 'distance': 776.1},
}


def run_benchmark(max_iterations: int = MAX_ITERATIONS,
                  use_parallel: bool = False,
                  n_parallel: int = N_PARALLEL_RUNS,
                  instances: List[str] = None):
    """Run ALNS on Solomon RC benchmark instances."""
    print("=" * 80)
    print("Enhanced ALNS for VRPTW - Solomon RC Benchmark")
    print("=" * 80)
    print(f"Settings: max_iter={max_iterations}, 2-opt=Yes, Noise={USE_NOISE}")
    if use_parallel:
        print(f"          Parallel runs: {n_parallel}")
    print()
    
    all_instances = load_rc_instances()
    
    if instances:
        all_instances = {k: v for k, v in all_instances.items() if k in instances}
    
    results = []
    
    for name, instance in sorted(all_instances.items()):
        print(f"\n{'='*60}")
        print(f"Instance: {name} ({instance.n_customers} customers)")
        print(f"{'='*60}")
        
        if use_parallel:
            solution, stats = parallel_alns(
                instance, 
                n_runs=n_parallel,
                max_iterations=max_iterations,
                verbose=True
            )
        else:
            solution, stats = alns_solve(
                instance, 
                max_iterations=max_iterations,
                verbose=True
            )
        
        bks = BKS.get(name, None)
        gap = 0.0
        if bks:
            gap = (solution.cost - bks['distance']) / bks['distance'] * 100
        
        print(f"\nFinal Result:")
        print(f"  Cost:      {solution.cost:.2f}")
        print(f"  Vehicles:  {solution.num_vehicles}")
        print(f"  Feasible:  {solution.feasible}")
        print(f"  Time:      {stats['solve_time']:.2f}s")
        print(f"  Iters:     {stats['iterations']}")
        if bks:
            print(f"  BKS:       {bks['distance']} ({bks['vehicles']} vehicles)")
            print(f"  Gap:       {gap:+.2f}%")
        
        results.append({
            'instance': name,
            'cost': solution.cost,
            'vehicles': solution.num_vehicles,
            'feasible': solution.feasible,
            'time': stats['solve_time'],
            'iterations': stats['iterations'],
            'bks_cost': bks['distance'] if bks else None,
            'bks_vehicles': bks['vehicles'] if bks else None,
            'gap': gap
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Instance':<10} {'Cost':>10} {'Veh':>5} {'BKS':>10} {'Gap':>8} {'Time':>8}")
    print("-" * 60)
    
    total_gap = 0.0
    for r in results:
        bks_str = f"{r['bks_cost']:.1f}" if r['bks_cost'] else "N/A"
        gap_str = f"{r['gap']:+.2f}%" if r['bks_cost'] else "N/A"
        print(f"{r['instance']:<10} {r['cost']:>10.2f} {r['vehicles']:>5} "
              f"{bks_str:>10} {gap_str:>8} {r['time']:>7.1f}s")
        if r['bks_cost']:
            total_gap += r['gap']
    
    avg_gap = total_gap / len([r for r in results if r['bks_cost']]) if results else 0
    print("-" * 60)
    print(f"Average Gap: {avg_gap:+.2f}%")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Enhanced ALNS Baseline Benchmark")
    print(f"Parameters: SIGMA1={SIGMA1}, SIGMA2={SIGMA2}, SIGMA3={SIGMA3}")
    print(f"            MAX_ITER={MAX_ITERATIONS}, ALPHA={ALPHA}")
    print(f"            USE_NOISE={USE_NOISE}, NOISE_PARAM={NOISE_PARAM}")
    print()
    
    # Run with 25000 iterations, 2-opt, and noise
    results = run_benchmark(
        max_iterations=MAX_ITERATIONS,
        use_parallel=False,  # Set True for parallel multi-start
        instances=None  # Run all instances
    )
    
    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
