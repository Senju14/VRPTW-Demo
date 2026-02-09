"""Hybrid DQN + ALNS solver (Proposed Framework)."""

import numpy as np
import torch
from typing import List

from ...schemas import VRPTWInstance, Solution
from ..baseline.alns import ALNSSolver
from .dqn_pointer import VRPTWPointerNetwork


class HybridSolver:
    """Hybrid solver using Attention Pointer Network warm-start and ALNS refinement."""
    
    def __init__(self, instance: VRPTWInstance, time_limit: float = 15.0):
        self.instance = instance
        self.time_limit = time_limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def solve(self) -> Solution:
        """Solve using RL construction + ALNS refinement."""
        try:
            # Note: In production, load the trained A100 model
            model = VRPTWPointerNetwork(input_dim=5, state_dim=4).to(self.device)
            # model.load_state_dict(torch.load("models/hybrid_v2_a100.pt", map_location=self.device))
            model.eval()
            initial_routes = self._construct_with_rl(model)
        except Exception as e:
            print(f"RL construction failed, falling back to pure ALNS: {e}")
            initial_routes = None
        
        alns = ALNSSolver(self.instance, self.time_limit)
        return alns.solve(initial_routes)
    
    def _construct_with_rl(self, model: VRPTWPointerNetwork) -> List[List[int]]:
        """Construct initial solution using the Attention Pointer Network."""
        inst = self.instance
        n = inst.n_customers
        
        # 1. Prepare normalized node features (B=1)
        features = np.zeros((n, 5), dtype=np.float32)
        for i, cust in enumerate(inst.customers):
            features[i, 0] = cust.x / inst.max_coord
            features[i, 1] = cust.y / inst.max_coord
            features[i, 2] = cust.demand / max(inst.vehicle_capacity, 1.0)
            features[i, 3] = cust.ready_time / inst.max_time
            features[i, 4] = cust.due_date / inst.max_time
        
        f_t = torch.tensor(features).unsqueeze(0).to(self.device)
        depot_xy = np.array([inst.depot.x / inst.max_coord, inst.depot.y / inst.max_coord], dtype=np.float32)
        
        routes = []
        unvisited = set(range(n))
        curr_route = []
        curr_node_idx = 0 # 0 is depot
        curr_node_xy = depot_xy
        curr_time = 0.0
        curr_load = 0.0
        
        while unvisited:
            # Build dynamic mask
            mask = np.ones(n, dtype=bool)
            for i in unvisited:
                cust = inst.customers[i]
                d = inst.distance(curr_node_idx, i + 1)
                arrival = max(curr_time + d, cust.ready_time)
                if arrival <= cust.due_date and curr_load + cust.demand <= inst.vehicle_capacity:
                    mask[i] = False
            
            if mask.all():
                # Back to depot, new vehicle
                if curr_route:
                    routes.append(curr_route)
                curr_route = []
                curr_node_idx = 0
                curr_node_xy = depot_xy
                curr_time = 0.0
                curr_load = 0.0
                continue
                
            # State: [x, y, load, time]
            state = np.array([curr_node_xy[0], curr_node_xy[1], curr_load / inst.vehicle_capacity, curr_time / inst.max_time], dtype=np.float32)
            s_t = torch.tensor(state).unsqueeze(0).to(self.device)
            m_t = torch.tensor(mask).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = model(f_t, s_t, m_t)
                action = logits.argmax(dim=-1).item()
                
            # Apply customer selection
            cust = inst.customers[action]
            node_id = action + 1
            d = inst.distance(curr_node_idx, node_id)
            arrival = max(curr_time + d, cust.ready_time)
            
            curr_time = arrival + cust.service_time
            curr_load += cust.demand
            curr_node_idx = node_id
            curr_node_xy = np.array([cust.x / inst.max_coord, cust.y / inst.max_coord])
            curr_route.append(node_id)
            unvisited.remove(action)
            
        if curr_route:
            routes.append(curr_route)
            
        return routes
