"""
DQN-only solver for VRPTW.
"""
from typing import Dict, List, Any

from src.backend.data_loader import Customer
from src.backend.solver import run_dqn_loop


def solve_dqn_only_vrptw(
    depot: Customer,
    customers: List[Customer],
    capacity: int,
    num_vehicles: int,
    model_path: str,
    iterations: int = 300,
) -> Dict[str, Any]:
    """Solve VRPTW using DQN-only (without ALNS)."""
    routes, cost, t = run_dqn_loop(
        depot,
        customers,
        capacity,
        num_vehicles,
        model_path,
        iterations,
        accept_prob=0.05,
    )
    return {"routes": routes, "total_distance": cost, "execution_time": t, "violations": []}

