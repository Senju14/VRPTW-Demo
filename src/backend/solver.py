import copy
import math
import random
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from src.backend.data_loader import Customer

# Import các hàm solver từ algorithms
from src.backend.algorithms.ortools_solver import solve_vrptw
from src.backend.algorithms.alns_solver import solve_alns_vrptw
from src.backend.algorithms.dqn_only_solver import solve_dqn_only_vrptw
from src.backend.algorithms.dqn_alns_solver import solve_dqn_alns_vrptw


# ---------------------------------------------------------------------------
# Shared utilities (dùng chung cho ALNS và DQN)
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Solution:
    """Solution representation for VRPTW."""

    def __init__(self, routes: List[List[Customer]]):
        self.routes = routes
        self.total_cost = 0.0
        self.total_time_violation = 0.0
        self.total_capacity_violation = 0.0


def calculate_distance(a: Customer, b: Customer) -> float:
    """Calculate Euclidean distance between two customers."""
    return math.hypot(a.x - b.x, a.y - b.y)


def evaluate_route(route: List[Customer], depot: Customer, capacity: int):
    """Evaluate a single route: cost, time violations, capacity violations."""
    cost = 0.0
    load = 0.0
    t = 0.0
    time_v = 0.0
    cap_v = 0.0
    last = depot

    for c in route:
        travel = calculate_distance(last, c)
        cost += travel
        arr = t + travel
        t = max(arr, c.ready_time)
        if t > c.due_date:
            time_v += t - c.due_date
        t += c.service_time
        load += c.demand
        last = c

    back = calculate_distance(last, depot)
    cost += back
    t += back
    if t > depot.due_date:
        time_v += t - depot.due_date
    if load > capacity:
        cap_v = load - capacity

    return cost, time_v, cap_v


def evaluate_solution(sol: Solution, depot: Customer, capacity: int):
    """Evaluate entire solution and update solution object."""
    total_cost = 0.0
    time_v = 0.0
    cap_v = 0.0
    routes: List[List[Customer]] = []

    for r in sol.routes:
        if not r:
            continue
        c, tv, cv = evaluate_route(r, depot, capacity)
        total_cost += c
        time_v += tv
        cap_v += cv
        routes.append(r)

    sol.routes = routes
    sol.total_cost = total_cost
    sol.total_time_violation = time_v
    sol.total_capacity_violation = cap_v


def create_initial_solution(
    depot: Customer,
    customers: List[Customer],
    capacity: int,
    max_vehicles: int | None,
):
    """Create initial feasible solution using greedy insertion."""
    routes: List[List[Customer]] = []
    pool = list(customers)
    random.shuffle(pool)
    unserved = set(pool)

    while unserved and (max_vehicles is None or len(routes) < max_vehicles):
        route: List[Customer] = []
        load = 0
        t = 0.0
        last = depot

        for c in list(unserved):
            travel = calculate_distance(last, c)
            arr = t + travel
            start = max(arr, c.ready_time)
            if load + c.demand <= capacity and start <= c.due_date:
                back = calculate_distance(c, depot)
                end = start + c.service_time
                finish = end + back
                if finish <= depot.due_date:
                    route.append(c)
                    unserved.remove(c)
                    load += c.demand
                    t = end
                    last = c

        if route:
            routes.append(route)
        if not route and unserved:
            break

    sol = Solution(routes)
    evaluate_solution(sol, depot, capacity)
    return sol


def random_removal(sol: Solution, depot: Customer, capacity: int, num_remove: int):
    """Destroy operator: randomly remove customers from solution."""
    all_customers: List[Customer] = [c for r in sol.routes for c in r]
    if not all_customers:
        return []

    num_remove = min(num_remove, len(all_customers))
    removed = random.sample(all_customers, num_remove)
    removed_set = set(removed)

    new_routes: List[List[Customer]] = []
    for r in sol.routes:
        keep = [c for c in r if c not in removed_set]
        if keep:
            new_routes.append(keep)

    sol.routes = new_routes
    return removed


def route_removal(sol: Solution, depot: Customer, capacity: int, num_routes: int):
    """Destroy operator: remove entire routes from solution."""
    if not sol.routes:
        return []

    num_routes = min(num_routes, len(sol.routes))
    sorted_routes = sorted(sol.routes, key=len)

    removed: List[Customer] = []
    kept: List[List[Customer]] = []
    for i, r in enumerate(sorted_routes):
        if i < num_routes:
            removed.extend(r)
        else:
            kept.append(r)

    sol.routes = kept
    return removed


def greedy_insertion(
    sol: Solution,
    to_insert: List[Customer],
    depot: Customer,
    capacity: int,
    max_vehicles: int | None,
):
    """Repair operator: greedily insert customers into solution."""
    random.shuffle(to_insert)

    for c in to_insert:
        best_r = -1
        best_pos = -1
        best_inc = float("inf")

        for i, r in enumerate(sol.routes):
            old_cost, _, _ = evaluate_route(r, depot, capacity)
            for p in range(len(r) + 1):
                cand = r[:p] + [c] + r[p:]
                new_cost, tv, cv = evaluate_route(cand, depot, capacity)
                if tv == 0 and cv == 0:
                    inc = new_cost - old_cost
                    if inc < best_inc:
                        best_inc = inc
                        best_r = i
                        best_pos = p

        if best_r != -1:
            sol.routes[best_r].insert(best_pos, c)
        else:
            if max_vehicles is None or len(sol.routes) < max_vehicles:
                cand = [c]
                _, tv, cv = evaluate_route(cand, depot, capacity)
                if tv == 0 and cv == 0:
                    sol.routes.append(cand)
            elif sol.routes:
                best_inc = float("inf")
                fb_r = 0
                fb_p = 0
                for i, r in enumerate(sol.routes):
                    old_cost, _, _ = evaluate_route(r, depot, capacity)
                    for p in range(len(r) + 1):
                        cand = r[:p] + [c] + r[p:]
                        new_cost, _, _ = evaluate_route(cand, depot, capacity)
                        inc = new_cost - old_cost
                        if inc < best_inc:
                            best_inc = inc
                            fb_r = i
                            fb_p = p
                sol.routes[fb_r].insert(fb_p, c)

    evaluate_solution(sol, depot, capacity)


# ---------------------------------------------------------------------------
# DQN Agent (dùng cho DQN-only và DQN+ALNS)
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    """Q-Network for DQN agent."""

    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.l1 = nn.Linear(state_size, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


class DQNAgent:
    """DQN Agent for selecting destroy operators."""

    def __init__(self, state_size: int, action_size: int):
        self.device = device
        self.model = QNetwork(state_size, action_size).to(self.device)

    def load_weights(self, path: str):
        """Load model weights from safetensors file."""
        state = load_file(path)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def select_action(self, state, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() > epsilon:
            with torch.no_grad():
                x = torch.FloatTensor([state]).to(self.device)
                q = self.model(x)
                return q.max(1)[1].item()
        return random.randint(0, 1)


def get_state(current_cost: float, best_cost: float, stuck: int):
    """Get state representation for DQN."""
    diff = (current_cost - best_cost) / best_cost if best_cost > 0 else 0.0
    s = stuck / 100.0
    return [diff, s]


def routes_to_indices(sol: Solution) -> List[List[int]]:
    """Convert solution routes to list of customer ID lists."""
    out: List[List[int]] = []
    for r in sol.routes:
        if not r:
            continue
        ids = [0] + [c.id for c in r]
        out.append(ids)
    return out


def run_dqn_loop(
    depot: Customer,
    customers: List[Customer],
    capacity: int,
    num_vehicles: int,
    model_path: str,
    iterations: int,
    accept_prob: float,
) -> Tuple[List[List[int]], float, float]:
    """Main DQN loop: uses DQN to select destroy operators."""
    destroy_ops = [random_removal, route_removal]
    agent = DQNAgent(2, 2)
    agent.load_weights(model_path)

    best = create_initial_solution(depot, customers, capacity, num_vehicles)
    cur = copy.deepcopy(best)
    stuck = 0
    start = time.time()

    for _ in range(1, iterations + 1):
        state = get_state(cur.total_cost, best.total_cost, stuck)
        idx = agent.select_action(state, epsilon=0.0)
        op = destroy_ops[idx]

        new = copy.deepcopy(cur)
        if op is random_removal:
            k = random.randint(5, 15)
            removed = op(new, depot, capacity, k)
        else:
            m = min(len(new.routes), max(1, len(new.routes) // 3))
            k = random.randint(1, m)
            removed = op(new, depot, capacity, k)

        greedy_insertion(new, removed, depot, capacity, num_vehicles)
        evaluate_solution(new, depot, capacity)

        if len(new.routes) <= num_vehicles:
            if new.total_cost < best.total_cost:
                best = copy.deepcopy(new)
                cur = copy.deepcopy(new)
                stuck = 0
            else:
                stuck += 1
                if random.random() < accept_prob:
                    cur = copy.deepcopy(new)
        else:
            stuck += 1

    elapsed = time.time() - start
    routes = routes_to_indices(best)
    return routes, best.total_cost, elapsed


# ---------------------------------------------------------------------------
# Public API - expose các hàm từ algorithms
# ---------------------------------------------------------------------------

__all__ = [
    "solve_vrptw",
    "solve_alns_vrptw",
    "solve_dqn_only_vrptw",
    "solve_dqn_alns_vrptw",
]
