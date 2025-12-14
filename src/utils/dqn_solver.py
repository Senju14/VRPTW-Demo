"""
Compatibility shim: solver functions are now split across smaller modules.
"""
from src.utils.solver_alns import solve_alns_vrptw
from src.utils.solver_dqn_only import solve_dqn_only_vrptw
from src.utils.solver_dqn_alns import solve_dqn_alns_vrptw, solve_dqn_vrptw

__all__ = [
    "solve_dqn_vrptw",
    "solve_dqn_alns_vrptw",
    "solve_dqn_only_vrptw",
    "solve_alns_vrptw",
]