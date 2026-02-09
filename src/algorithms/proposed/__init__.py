"""
Proposed algorithms for VRPTW optimization.

Modules:
- dqn_pointer: Attention-based Pointer Network for construction
- hybrid: RL construction + ALNS refinement
- dqn_alns: GNN-D3QN guided ALNS operator selection
- instance_generator: Synthetic Solomon-style instances
"""

from .dqn_pointer import VRPTWPointerNetwork
from .hybrid import HybridSolver

# DQN-ALNS components
from .dqn_alns import (
    DQNConfig,
    GATEncoder,
    DuelingDQN,
    DQNALNSSolver,
    ParallelDQNALNS,
    PrioritizedReplayBuffer,
)

from .instance_generator import (
    generate_solomon_style_instance,
    generate_training_batch,
    generate_scaling_test_instances,
)

__all__ = [
    "VRPTWPointerNetwork",
    "HybridSolver",
    "DQNConfig",
    "GATEncoder", 
    "DuelingDQN",
    "DQNALNSSolver",
    "ParallelDQNALNS",
    "PrioritizedReplayBuffer",
    "generate_solomon_style_instance",
    "generate_training_batch",
    "generate_scaling_test_instances",
]
