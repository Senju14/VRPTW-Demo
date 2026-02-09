"""
DQN-ALNS for VRPTW — Hierarchical Objective Model
===================================================
Based on the ALNS baseline (Ropke & Pisinger 2006) with the following key
refactors:

1.  **Hierarchical Objective (Big-M Penalty)**
    Internal_Cost = Total_Distance + W * Num_Vehicles
    where W = 10 000 so that the solver strictly minimises fleet size first,
    then total distance.

2.  **Decoupled Logging & Reporting**
    The inflated Internal_Cost is *never* printed.  All logs, progress bars,
    and final reports show Real_Distance and Num_Vehicles separately.

3.  **Hierarchical Gap-to-BKS**
    A solution with fewer vehicles is always considered better, regardless
    of distance.  When vehicle counts are equal, distances are compared.

All existing 2-opt / Or-opt local search, noise logic, and parallel
execution structures are preserved from the baseline.
"""

from __future__ import annotations

import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# HYPERPARAMETERS (from Ropke & Pisinger 2006)
# =============================================================================

# --- Adaptive Weight Update ---
SIGMA1: int = 33       # New global best solution found
SIGMA2: int = 20       # Better than current solution
SIGMA3: int = 13       # Accepted by SA (worse, but accepted)
RHO: float = 0.1       # Reaction factor for weight updates
ETA_S: int = 100       # Weight update period (iterations)

# --- Simulated Annealing ---
ALPHA: float = 0.99975  # Cooling rate
W_START: float = 0.05   # Initial temperature parameter

# --- Removal Parameters ---
Q_MIN: int = 4          # Min customers to remove
Q_MAX: int = 100        # Max (capped at 40 % of customers)

# --- Shaw Removal (Relatedness) ---
PHI: int = 9            # Distance weight
CHI: int = 3            # Time weight
PSI: int = 2            # Demand weight
P_SHAW: int = 6         # Randomness (higher → more deterministic)

# --- Worst Removal ---
P_WORST: int = 3        # Randomness

# --- Noise (Ropke & Pisinger) ---
NOISE_PARAM: float = 0.025
USE_NOISE: bool = True

# --- Stopping Criteria ---
MAX_ITERATIONS: int = 25_000
MAX_NO_IMPROVE: int = 5_000   # Early stop if no improvement

# --- Parallel Multi-Start ---
N_PARALLEL_RUNS: int = 4

# =============================================================================
# BIG-M WEIGHT — Hierarchical Objective
# =============================================================================
# A sufficiently large constant so that removing one vehicle always outweighs
# any feasible distance improvement.  For Solomon 100-customer instances the
# maximum total distance never exceeds ~2 500, so W = 10 000 is safe.
VEHICLE_WEIGHT: float = 10_000.0


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
    num_vehicles: int                                  # max fleet in data file
    dist_matrix: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._build_distance_matrix()
        all_nodes: List[Customer] = [self.depot] + self.customers
        self.max_coord: float = max(
            max(c.x for c in all_nodes), max(c.y for c in all_nodes)
        )
        self.max_time: float = max(c.due_date for c in all_nodes)
        self.max_demand: float = max(
            (c.demand for c in self.customers), default=1.0
        )
        self.max_distance: float = float(np.max(self.dist_matrix))

    # --------------------------------------------------------------------- #
    def _build_distance_matrix(self) -> None:
        """Pre-compute Euclidean distance matrix for all nodes."""
        n: int = len(self.customers) + 1
        self.dist_matrix = np.zeros((n, n), dtype=np.float64)
        all_nodes: List[Customer] = [self.depot] + self.customers
        for i, n1 in enumerate(all_nodes):
            for j, n2 in enumerate(all_nodes):
                self.dist_matrix[i, j] = math.sqrt(
                    (n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2
                )

    def distance(self, i: int, j: int) -> float:
        """Distance between nodes *i* and *j* (0 = depot)."""
        return float(self.dist_matrix[i, j])

    def get_customer(self, cid: int) -> Customer:
        """Return customer by ID (1-indexed)."""
        return self.customers[cid - 1]

    @property
    def n_customers(self) -> int:
        return len(self.customers)


@dataclass
class Solution:
    """
    A VRPTW solution with **hierarchical cost**.

    *   ``cost``         — internal penalised cost used ONLY by SA acceptance
                           and best-solution tracking.
    *   ``real_distance`` — true total Euclidean distance (for display).
    *   ``num_vehicles``  — number of active routes (for display).
    """
    routes: List[List[int]]
    instance: VRPTWInstance
    cost: float = field(default=0.0, init=False)
    real_distance: float = field(default=0.0, init=False)
    feasible: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        self.real_distance = self._compute_distance()
        self.cost = self._compute_cost()
        self.feasible = self._check_feasibility()

    # ------------------------------------------------------------------ #
    #  Hierarchical cost: distance + W × vehicles
    # ------------------------------------------------------------------ #
    def _compute_distance(self) -> float:
        """Total Euclidean distance across all routes (no penalty)."""
        total: float = 0.0
        for route in self.routes:
            if not route:
                continue
            total += self.instance.distance(0, route[0])
            for i in range(len(route) - 1):
                total += self.instance.distance(route[i], route[i + 1])
            total += self.instance.distance(route[-1], 0)
        return total

    def _compute_cost(self) -> float:
        """
        Internal cost with Big-M penalty for fleet size.

        Formula:
            Internal_Cost = Total_Distance + VEHICLE_WEIGHT × Num_Vehicles

        This ensures that reducing one vehicle is always preferred over
        any distance improvement, aligning with the standard Solomon
        hierarchical objective.
        """
        return self.real_distance + VEHICLE_WEIGHT * self.num_vehicles

    # ------------------------------------------------------------------ #
    def _check_feasibility(self) -> bool:
        """Check capacity and time window constraints for every route."""
        inst = self.instance
        for route in self.routes:
            if not route:
                continue
            # Capacity
            load: float = sum(inst.get_customer(cid).demand for cid in route)
            if load > inst.vehicle_capacity:
                return False
            # Time windows
            t: float = 0.0
            prev: int = 0
            for cid in route:
                cust = inst.get_customer(cid)
                t += inst.distance(prev, cid)
                t = max(t, cust.ready_time)
                if t > cust.due_date:
                    return False
                t += cust.service_time
                prev = cid
        return True

    # ------------------------------------------------------------------ #
    @property
    def num_vehicles(self) -> int:
        """Number of non-empty routes (active vehicles)."""
        return len([r for r in self.routes if r])

    def copy(self) -> "Solution":
        return Solution([r[:] for r in self.routes], self.instance)

    def recalculate(self) -> None:
        """Re-compute distance, cost, and feasibility after in-place edits."""
        self.real_distance = self._compute_distance()
        self.cost = self._compute_cost()
        self.feasible = self._check_feasibility()


# =============================================================================
# HIERARCHICAL COMPARISON HELPERS
# =============================================================================

def is_better(a: Solution, b: Solution) -> bool:
    """
    Return ``True`` if *a* is strictly better than *b* under the
    hierarchical objective.

    With the Big-M penalised ``cost`` this is simply ``a.cost < b.cost``,
    but we keep the function explicit for clarity and future-proofing.
    """
    return a.cost < b.cost


def hierarchical_gap(
    sol_vehicles: int,
    sol_distance: float,
    bks_vehicles: int,
    bks_distance: float,
) -> float:
    """
    Compute gap-to-BKS respecting the hierarchy:
    *   Fewer vehicles → always better (negative gap = better).
    *   More vehicles  → always worse  (positive gap = worse).
    *   Equal vehicles → compare distances (percentage gap).
    """
    if sol_vehicles < bks_vehicles:
        # Strictly better on the primary objective
        return -100.0 * (bks_vehicles - sol_vehicles)  # large negative %
    if sol_vehicles > bks_vehicles:
        # Strictly worse on the primary objective
        return 100.0 * (sol_vehicles - bks_vehicles)   # large positive %
    # Same fleet size → compare distances
    if bks_distance == 0:
        return 0.0
    return (sol_distance - bks_distance) / bks_distance * 100.0


# =============================================================================
# DATA LOADING
# =============================================================================

def parse_solomon_file(filepath: str) -> VRPTWInstance:
    """Parse a Solomon-format benchmark file."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    name: str = lines[0].strip()
    vehicle_line = lines[4].split()
    num_vehicles: int = int(vehicle_line[0])
    capacity: float = float(vehicle_line[1])

    nodes: List[dict] = []
    for line in lines[9:]:
        parts = line.split()
        if len(parts) >= 7:
            nodes.append(
                {
                    "id": int(parts[0]),
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "demand": float(parts[3]),
                    "ready_time": float(parts[4]),
                    "due_date": float(parts[5]),
                    "service_time": float(parts[6]),
                }
            )

    depot_data = nodes[0]
    depot = Customer(
        id=0,
        x=depot_data["x"],
        y=depot_data["y"],
        demand=0,
        ready_time=depot_data["ready_time"],
        due_date=depot_data["due_date"],
        service_time=0,
    )
    customers = [
        Customer(
            id=n["id"],
            x=n["x"],
            y=n["y"],
            demand=n["demand"],
            ready_time=n["ready_time"],
            due_date=n["due_date"],
            service_time=n["service_time"],
        )
        for n in nodes[1:]
    ]

    return VRPTWInstance(
        name=name,
        depot=depot,
        customers=customers,
        vehicle_capacity=capacity,
        num_vehicles=num_vehicles,
    )


def get_solomon_data_path() -> str:
    """Locate the Solomon benchmark data directory."""
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
    instances: Dict[str, VRPTWInstance] = {}

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
    """Construct an initial solution using a Nearest-Neighbour heuristic."""
    routes: List[List[int]] = []
    unvisited: set = set(range(1, instance.n_customers + 1))

    while unvisited:
        route: List[int] = []
        curr_node: int = 0
        curr_time: float = 0.0
        curr_load: float = 0.0

        while True:
            best_cust: Optional[int] = None
            best_dist: float = float("inf")

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
            curr_time = (
                max(curr_time + best_dist, cust.ready_time) + cust.service_time
            )
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
    """Check if a single route satisfies capacity and time windows."""
    if not route:
        return True

    load: float = sum(instance.get_customer(cid).demand for cid in route)
    if load > instance.vehicle_capacity:
        return False

    t: float = 0.0
    prev: int = 0
    for cid in route:
        cust = instance.get_customer(cid)
        t += instance.distance(prev, cid)
        t = max(t, cust.ready_time)
        if t > cust.due_date:
            return False
        t += cust.service_time
        prev = cid

    if t + instance.distance(route[-1], 0) > instance.depot.due_date:
        return False

    return True


def route_cost(route: List[int], instance: VRPTWInstance) -> float:
    """Total Euclidean distance of a single route (depot → … → depot)."""
    if not route:
        return 0.0
    cost: float = instance.distance(0, route[0])
    for i in range(len(route) - 1):
        cost += instance.distance(route[i], route[i + 1])
    cost += instance.distance(route[-1], 0)
    return cost


# ------------------------------------------------------------------ #
# 2-opt intra-route
# ------------------------------------------------------------------ #

def two_opt_route(route: List[int], instance: VRPTWInstance) -> List[int]:
    """2-opt within a single route: reverse segments to reduce distance."""
    if len(route) < 4:
        return route

    improved: bool = True
    best_route: List[int] = route[:]
    best_cost: float = route_cost(best_route, instance)

    while improved:
        improved = False
        for i in range(len(best_route) - 2):
            for j in range(i + 2, len(best_route)):
                new_route = (
                    best_route[: i + 1]
                    + best_route[i + 1 : j + 1][::-1]
                    + best_route[j + 1 :]
                )
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
    """Apply 2-opt to every route in the solution."""
    sol = solution.copy()
    for i in range(len(sol.routes)):
        if len(sol.routes[i]) >= 4:
            sol.routes[i] = two_opt_route(sol.routes[i], sol.instance)
    sol.recalculate()
    return sol


# ------------------------------------------------------------------ #
# Or-opt intra-route
# ------------------------------------------------------------------ #

def or_opt_route(route: List[int], instance: VRPTWInstance) -> List[int]:
    """Or-opt: relocate 1–3 consecutive customers to a better position."""
    if len(route) < 3:
        return route

    improved: bool = True
    best_route: List[int] = route[:]
    best_cost: float = route_cost(best_route, instance)

    while improved:
        improved = False
        for seg_size in [1, 2, 3]:
            if seg_size > len(best_route):
                continue
            for i in range(len(best_route) - seg_size + 1):
                segment = best_route[i : i + seg_size]
                remaining = best_route[:i] + best_route[i + seg_size :]
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
    """Apply Or-opt then 2-opt to every route."""
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

def random_removal(
    solution: Solution, q: int
) -> Tuple[Solution, List[int]]:
    """Remove *q* random customers."""
    sol = solution.copy()
    all_custs: List[int] = [c for r in sol.routes for c in r]

    if len(all_custs) < q:
        q = len(all_custs)

    removed: List[int] = random.sample(all_custs, q)

    for route in sol.routes:
        for c in removed:
            if c in route:
                route.remove(c)

    sol.routes = [r for r in sol.routes if r]
    sol.recalculate()
    return sol, removed


def shaw_removal(
    solution: Solution, q: int
) -> Tuple[Solution, List[int]]:
    """
    Shaw Removal (Algorithm 2, Ropke & Pisinger 2006).
    Remove related requests based on distance, time, and demand similarity.
    """
    sol = solution.copy()
    inst = sol.instance
    all_custs: List[int] = [c for r in sol.routes for c in r]

    if not all_custs:
        return sol, []

    q = min(q, len(all_custs))
    seed: int = random.choice(all_custs)
    D: List[int] = [seed]

    while len(D) < q:
        r = random.choice(D)
        r_cust = inst.get_customer(r)

        L: List[int] = [c for c in all_custs if c not in D]
        if not L:
            break

        relatedness: List[Tuple[int, float]] = []
        for cid in L:
            cust = inst.get_customer(cid)
            dist = inst.distance(r, cid) / inst.max_coord
            time_diff = abs(r_cust.ready_time - cust.ready_time) / inst.max_time
            demand_diff = abs(r_cust.demand - cust.demand) / max(
                inst.max_demand, 1.0
            )
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


def worst_removal(
    solution: Solution, q: int
) -> Tuple[Solution, List[int]]:
    """
    Worst Removal (Algorithm 3, Ropke & Pisinger 2006).
    Remove one request at a time, re-evaluating after each removal.
    """
    sol = solution.copy()
    inst = sol.instance
    removed: List[int] = []

    while len(removed) < q:
        costs: List[Tuple[int, float, int]] = []
        for route_idx, route in enumerate(sol.routes):
            for pos, cid in enumerate(route):
                prev_node = route[pos - 1] if pos > 0 else 0
                next_node = route[pos + 1] if pos < len(route) - 1 else 0
                saving = (
                    inst.distance(prev_node, cid)
                    + inst.distance(cid, next_node)
                    - inst.distance(prev_node, next_node)
                )
                costs.append((cid, saving, route_idx))

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

def calc_insertion_cost(
    route: List[int],
    cid: int,
    pos: int,
    instance: VRPTWInstance,
    use_noise: bool = False,
) -> float:
    """
    Insertion cost with optional noise.  Returns ``inf`` when infeasible.
    """
    cust = instance.get_customer(cid)

    # Capacity check
    load: float = sum(instance.get_customer(c).demand for c in route) + cust.demand
    if load > instance.vehicle_capacity:
        return float("inf")

    # Time-window feasibility
    test_route: List[int] = route[:pos] + [cid] + route[pos:]
    t: float = 0.0
    prev: int = 0
    for node_id in test_route:
        node = instance.get_customer(node_id)
        t += instance.distance(prev, node_id)
        t = max(t, node.ready_time)
        if t > node.due_date:
            return float("inf")
        t += node.service_time
        prev = node_id
    if t + instance.distance(prev, 0) > instance.depot.due_date:
        return float("inf")

    # Pure distance increase
    prev_node = route[pos - 1] if pos > 0 else 0
    next_node = route[pos] if pos < len(route) else 0
    cost_increase: float = (
        instance.distance(prev_node, cid)
        + instance.distance(cid, next_node)
        - instance.distance(prev_node, next_node)
    )

    # Noise
    if use_noise and USE_NOISE:
        noise = NOISE_PARAM * instance.max_distance * random.uniform(-1, 1)
        cost_increase = max(0, cost_increase + noise)

    return cost_increase


def find_best_insertion(
    routes: List[List[int]],
    cid: int,
    instance: VRPTWInstance,
    use_noise: bool = False,
) -> Tuple[int, int, float]:
    """Find the cheapest feasible insertion position for *cid*."""
    best_route: int = -1
    best_pos: int = -1
    best_cost: float = float("inf")

    for r_idx, route in enumerate(routes):
        for pos in range(len(route) + 1):
            cost = calc_insertion_cost(route, cid, pos, instance, use_noise)
            if cost < best_cost:
                best_cost = cost
                best_route = r_idx
                best_pos = pos

    return best_route, best_pos, best_cost


def greedy_insertion(
    solution: Solution, removed: List[int], use_noise: bool = False
) -> Solution:
    """Greedy insertion with optional noise."""
    sol = solution.copy()
    inst = sol.instance
    random.shuffle(removed)

    for cid in removed:
        r_idx, pos, _ = find_best_insertion(sol.routes, cid, inst, use_noise)
        if r_idx != -1:
            sol.routes[r_idx].insert(pos, cid)
        else:
            # Open a new route (new vehicle)
            sol.routes.append([cid])

    sol.recalculate()
    return sol


def regret_k_insertion(
    solution: Solution,
    removed: List[int],
    k: int = 2,
    use_noise: bool = False,
) -> Solution:
    """
    Regret-*k* insertion: prioritise customers with the highest regret
    (difference between best and k-th best insertion cost).
    """
    sol = solution.copy()
    inst = sol.instance
    remaining: List[int] = removed[:]

    while remaining:
        regret_values: List[Tuple[int, float, Tuple[int, int, float]]] = []

        for cid in remaining:
            all_costs: List[Tuple[int, int, float]] = []

            for r_idx, route in enumerate(sol.routes):
                for pos in range(len(route) + 1):
                    cost = calc_insertion_cost(route, cid, pos, inst, use_noise)
                    if cost < float("inf"):
                        all_costs.append((r_idx, pos, cost))

            # Option: open a new route
            new_route_cost = calc_insertion_cost([], cid, 0, inst, use_noise)
            if new_route_cost < float("inf"):
                all_costs.append((len(sol.routes), 0, new_route_cost))

            if not all_costs:
                all_costs.append((len(sol.routes), 0, 0))

            all_costs.sort(key=lambda x: x[2])
            best = all_costs[0]
            regret: float = 0.0
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
    """Regret-2 wrapper."""
    return regret_k_insertion(solution, removed, k=2, use_noise=USE_NOISE)


def regret_3_insertion(solution: Solution, removed: List[int]) -> Solution:
    """Regret-3 wrapper."""
    return regret_k_insertion(solution, removed, k=3, use_noise=USE_NOISE)


# =============================================================================
# ADAPTIVE WEIGHT MECHANISM
# =============================================================================

class AdaptiveWeights:
    """Manages roulette-wheel adaptive weights for operator selection."""

    def __init__(self, n_destroy: int, n_repair: int) -> None:
        self.w_destroy: np.ndarray = np.ones(n_destroy)
        self.w_repair: np.ndarray = np.ones(n_repair)

        self.scores_d: np.ndarray = np.zeros(n_destroy)
        self.scores_r: np.ndarray = np.zeros(n_repair)
        self.counts_d: np.ndarray = np.zeros(n_destroy)
        self.counts_r: np.ndarray = np.zeros(n_repair)

    def select_operator(self, weights: np.ndarray) -> int:
        """Roulette-wheel selection based on current weights."""
        probs = weights / weights.sum()
        return int(np.random.choice(len(weights), p=probs))

    def update_score(self, d_idx: int, r_idx: int, score: float) -> None:
        self.scores_d[d_idx] += score
        self.scores_r[r_idx] += score
        self.counts_d[d_idx] += 1
        self.counts_r[r_idx] += 1

    def update_weights(self) -> None:
        """Update weights at the end of each reaction period."""
        for i in range(len(self.w_destroy)):
            if self.counts_d[i] > 0:
                avg = self.scores_d[i] / self.counts_d[i]
                self.w_destroy[i] = (1 - RHO) * self.w_destroy[i] + RHO * avg

        for i in range(len(self.w_repair)):
            if self.counts_r[i] > 0:
                avg = self.scores_r[i] / self.counts_r[i]
                self.w_repair[i] = (1 - RHO) * self.w_repair[i] + RHO * avg

        self.scores_d.fill(0)
        self.scores_r.fill(0)
        self.counts_d.fill(0)
        self.counts_r.fill(0)


# =============================================================================
# SIMULATED ANNEALING ACCEPTANCE
# =============================================================================

def init_temperature(init_cost: float, w: float = W_START) -> float:
    """Initial SA temperature calibrated so that a *w*-worse solution is
    accepted with probability 0.5."""
    return -w * init_cost / math.log(0.5)


def accept_solution(
    cost_new: float, cost_current: float, temperature: float
) -> bool:
    """Simulated Annealing acceptance criterion (uses Internal_Cost)."""
    if cost_new < cost_current:
        return True
    if temperature <= 0:
        return False
    delta: float = cost_new - cost_current
    probability: float = math.exp(-delta / max(temperature, 1e-300))
    return random.random() < probability


# =============================================================================
# MAIN ALNS ALGORITHM
# =============================================================================

def alns_solve(
    instance: VRPTWInstance,
    max_iterations: int = MAX_ITERATIONS,
    use_local_search: bool = True,
    verbose: bool = True,
    seed: int | None = None,
) -> Tuple[Solution, Dict]:
    """
    Enhanced ALNS with Hierarchical Objective.

    *   Iteration-based stopping
    *   2-opt / Or-opt local search
    *   Noise in insertion cost
    *   Big-M penalised cost for SA acceptance & best-tracking
    *   Decoupled display: Real_Distance + Num_Vehicles (never Internal_Cost)

    Returns
    -------
    best_solution : Solution
    stats : dict
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ---- Operators ----
    destroy_ops: List[Callable] = [random_removal, shaw_removal, worst_removal]
    repair_ops: List[Callable] = [greedy_insertion, regret_2_insertion, regret_3_insertion]
    destroy_names: List[str] = ["Random", "Shaw", "Worst"]
    repair_names: List[str] = ["Greedy", "Regret-2", "Regret-3"]

    # ---- Initial solution ----
    S: Solution = nearest_neighbor_construction(instance)
    if use_local_search:
        S = local_search_2opt(S)

    S_best: Solution = S.copy()
    S_current: Solution = S.copy()

    # SA temperature is calibrated on the *penalised* Internal_Cost
    T: float = init_temperature(S.cost)

    # Adaptive weights
    weights = AdaptiveWeights(len(destroy_ops), len(repair_ops))

    # ---- Statistics (display-friendly: real_distance, not internal cost) ----
    stats: Dict = {
        "iterations": 0,
        "improvements": 0,
        "accepts": 0,
        "init_distance": S.real_distance,
        "init_vehicles": S.num_vehicles,
    }

    start_time: float = time.time()
    no_improve_count: int = 0

    for iteration in range(1, max_iterations + 1):
        # -- Early termination --
        if no_improve_count >= MAX_NO_IMPROVE:
            if verbose:
                print(
                    f"  Early stop: no improvement for {MAX_NO_IMPROVE} iterations"
                )
            break

        # 1. Select operators (roulette wheel)
        d_idx: int = weights.select_operator(weights.w_destroy)
        r_idx: int = weights.select_operator(weights.w_repair)
        destroy_op = destroy_ops[d_idx]
        repair_op = repair_ops[r_idx]

        # 2. Decide removal size (4 … 40 % of customers)
        n_custs: int = sum(len(r) for r in S_current.routes)
        q_max: int = max(Q_MIN, int(0.4 * n_custs))
        q: int = random.randint(Q_MIN, min(Q_MAX, q_max))

        # 3. Destroy → Repair
        S_destroyed, removed = destroy_op(S_current, q)
        S_new: Solution = repair_op(S_destroyed, removed)

        # 4. Local search only when a new global best is found
        if use_local_search and is_better(S_new, S_best):
            S_new = local_search_2opt(S_new)

        # 5. Evaluate — use Internal_Cost for SA acceptance
        score: float = 0

        if is_better(S_new, S_best):
            # New global best
            S_best = S_new.copy()
            S_current = S_new
            score = SIGMA1
            stats["improvements"] += 1
            no_improve_count = 0

            # -- Decoupled display: Real_Distance + Vehicles, never cost --
            if verbose and (
                iteration <= 100
                or iteration % 100 == 0
                or stats["improvements"] <= 50
            ):
                elapsed = time.time() - start_time
                print(
                    f"  Iter {iteration:>6d}: ★ New best  "
                    f"Dist={S_best.real_distance:>9.2f}  "
                    f"Veh={S_best.num_vehicles:>3d}  "
                    f"@ {elapsed:.1f}s"
                )

        elif S_new.cost < S_current.cost:
            # Better than current (SA always accepts)
            S_current = S_new
            score = SIGMA2
            stats["accepts"] += 1
            no_improve_count = 0

        elif accept_solution(S_new.cost, S_current.cost, T):
            # Accepted by SA (worse but within threshold)
            S_current = S_new
            score = SIGMA3
            stats["accepts"] += 1
            no_improve_count += 1
        else:
            no_improve_count += 1

        # 6. Adaptive weight update
        weights.update_score(d_idx, r_idx, score)
        if iteration % ETA_S == 0:
            weights.update_weights()

        # 7. Cool temperature
        T *= ALPHA

    # ---- Final local search on best solution ----
    if use_local_search:
        S_best = local_search_full(S_best)

    stats["iterations"] = iteration
    stats["final_distance"] = S_best.real_distance
    stats["final_vehicles"] = S_best.num_vehicles
    stats["solve_time"] = time.time() - start_time
    stats["destroy_weights"] = dict(
        zip(destroy_names, weights.w_destroy.tolist())
    )
    stats["repair_weights"] = dict(
        zip(repair_names, weights.w_repair.tolist())
    )

    return S_best, stats


# =============================================================================
# PARALLEL MULTI-START
# =============================================================================

def _single_run_wrapper(args: tuple) -> Tuple[Solution, Dict]:
    """Pickle-friendly wrapper for ``ProcessPoolExecutor``."""
    instance, max_iterations, use_local_search, run_seed = args
    return alns_solve(
        instance,
        max_iterations=max_iterations,
        use_local_search=use_local_search,
        verbose=False,
        seed=run_seed,
    )


def parallel_alns(
    instance: VRPTWInstance,
    n_runs: int = N_PARALLEL_RUNS,
    max_iterations: int = MAX_ITERATIONS,
    use_local_search: bool = True,
    verbose: bool = True,
) -> Tuple[Solution, Dict]:
    """
    Run *n_runs* independent ALNS instances in parallel and return the
    best result under the hierarchical objective.
    """
    if verbose:
        print(f"  Running {n_runs} parallel ALNS instances …")

    start_time: float = time.time()

    args_list = [
        (instance, max_iterations, use_local_search, random.randint(0, 10_000) + i)
        for i in range(n_runs)
    ]

    with ProcessPoolExecutor(max_workers=n_runs) as executor:
        results = list(executor.map(_single_run_wrapper, args_list))

    # Pick the best using the hierarchical (penalised) cost
    best_solution: Optional[Solution] = None
    best_cost: float = float("inf")
    best_stats: Dict = {}
    total_iterations: int = 0

    for solution, stats in results:
        total_iterations += stats["iterations"]
        if solution.cost < best_cost:
            best_cost = solution.cost
            best_solution = solution
            best_stats = stats

    if verbose and best_solution is not None:
        elapsed = time.time() - start_time
        print(
            f"  Parallel complete:  Dist={best_solution.real_distance:>9.2f}  "
            f"Veh={best_solution.num_vehicles:>3d}  in {elapsed:.1f}s"
        )

    best_stats["total_time"] = time.time() - start_time
    best_stats["total_iterations"] = total_iterations

    return best_solution, best_stats  # type: ignore[return-value]


# =============================================================================
# BENCHMARK (Best Known Solutions for Solomon RC instances)
# =============================================================================

BKS: Dict[str, Dict[str, float]] = {
    "RC101": {"vehicles": 14, "distance": 1619.8},
    "RC102": {"vehicles": 12, "distance": 1457.4},
    "RC103": {"vehicles": 11, "distance": 1258.0},
    "RC104": {"vehicles": 10, "distance": 1132.3},
    "RC105": {"vehicles": 13, "distance": 1513.7},
    "RC106": {"vehicles": 11, "distance": 1372.7},
    "RC107": {"vehicles": 11, "distance": 1207.8},
    "RC108": {"vehicles": 10, "distance": 1114.2},
    "RC201": {"vehicles": 4,  "distance": 1261.8},
    "RC202": {"vehicles": 3,  "distance": 1092.3},
    "RC203": {"vehicles": 3,  "distance": 923.7},
    "RC204": {"vehicles": 3,  "distance": 783.5},
    "RC205": {"vehicles": 4,  "distance": 1154.0},
    "RC206": {"vehicles": 3,  "distance": 1051.1},
    "RC207": {"vehicles": 3,  "distance": 962.9},
    "RC208": {"vehicles": 3,  "distance": 776.1},
}


def run_benchmark(
    max_iterations: int = MAX_ITERATIONS,
    use_parallel: bool = False,
    n_parallel: int = N_PARALLEL_RUNS,
    instances: Optional[List[str]] = None,
) -> List[Dict]:
    """Run DQN-ALNS on Solomon RC benchmark instances."""
    print("=" * 80)
    print("DQN-ALNS for VRPTW — Hierarchical Objective (Solomon RC Benchmark)")
    print("=" * 80)
    print(
        f"Settings: max_iter={max_iterations}, 2-opt=Yes, Noise={USE_NOISE}, "
        f"W={VEHICLE_WEIGHT:.0f}"
    )
    if use_parallel:
        print(f"          Parallel runs: {n_parallel}")
    print()

    all_instances = load_rc_instances()
    if instances:
        all_instances = {k: v for k, v in all_instances.items() if k in instances}

    results: List[Dict] = []

    for name, instance in sorted(all_instances.items()):
        print(f"\n{'=' * 60}")
        print(f"Instance: {name} ({instance.n_customers} customers)")
        print(f"{'=' * 60}")

        if use_parallel:
            solution, stats = parallel_alns(
                instance,
                n_runs=n_parallel,
                max_iterations=max_iterations,
                verbose=True,
            )
        else:
            solution, stats = alns_solve(
                instance,
                max_iterations=max_iterations,
                verbose=True,
            )

        # ---- Hierarchical gap computation ----
        bks = BKS.get(name)
        gap: float = 0.0
        if bks:
            gap = hierarchical_gap(
                sol_vehicles=solution.num_vehicles,
                sol_distance=solution.real_distance,
                bks_vehicles=int(bks["vehicles"]),
                bks_distance=bks["distance"],
            )

        # ---- Decoupled final report (no internal cost shown) ----
        print(f"\nFinal Result:")
        print(f"  Distance:  {solution.real_distance:.2f}")
        print(f"  Vehicles:  {solution.num_vehicles}")
        print(f"  Feasible:  {solution.feasible}")
        print(f"  Time:      {stats['solve_time']:.2f}s")
        print(f"  Iters:     {stats['iterations']}")
        if bks:
            print(
                f"  BKS:       {bks['distance']} "
                f"({int(bks['vehicles'])} vehicles)"
            )
            print(f"  Gap:       {gap:+.2f}%")

        results.append(
            {
                "instance": name,
                "distance": solution.real_distance,
                "vehicles": solution.num_vehicles,
                "feasible": solution.feasible,
                "time": stats["solve_time"],
                "iterations": stats["iterations"],
                "bks_distance": bks["distance"] if bks else None,
                "bks_vehicles": int(bks["vehicles"]) if bks else None,
                "gap": gap,
            }
        )

    # ---- Summary table (decoupled: distance + vehicles separately) ----
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"{'Instance':<10} {'Distance':>10} {'Veh':>5} "
        f"{'BKS Dist':>10} {'BKS V':>6} {'Gap':>8} {'Time':>8}"
    )
    print("-" * 68)

    total_gap: float = 0.0
    n_with_bks: int = 0

    for r in results:
        bks_d_str = f"{r['bks_distance']:.1f}" if r["bks_distance"] else "N/A"
        bks_v_str = str(r["bks_vehicles"]) if r["bks_vehicles"] else "N/A"
        gap_str = f"{r['gap']:+.2f}%" if r["bks_distance"] else "N/A"

        print(
            f"{r['instance']:<10} {r['distance']:>10.2f} {r['vehicles']:>5} "
            f"{bks_d_str:>10} {bks_v_str:>6} {gap_str:>8} {r['time']:>7.1f}s"
        )

        if r["bks_distance"]:
            total_gap += r["gap"]
            n_with_bks += 1

    avg_gap: float = total_gap / n_with_bks if n_with_bks else 0.0
    print("-" * 68)
    print(f"Average Gap: {avg_gap:+.2f}%")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("DQN-ALNS — Hierarchical Objective (Fleet Size → Distance)")
    print(f"Parameters: SIGMA1={SIGMA1}, SIGMA2={SIGMA2}, SIGMA3={SIGMA3}")
    print(f"            MAX_ITER={MAX_ITERATIONS}, ALPHA={ALPHA}")
    print(f"            USE_NOISE={USE_NOISE}, NOISE_PARAM={NOISE_PARAM}")
    print(f"            VEHICLE_WEIGHT={VEHICLE_WEIGHT:.0f}")
    print()

    results = run_benchmark(
        max_iterations=MAX_ITERATIONS,
        use_parallel=False,   # Set True for parallel multi-start
        instances=None,       # Run all RC instances
    )

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
