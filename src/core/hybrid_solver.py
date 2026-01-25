"""Hybrid DQN + ALNS solver for VRPTW."""

import numpy as np
import torch
from typing import List

from .vrptw_types import VRPTWInstance, Solution
from .alns_solver import ALNSSolver
from .dqn_model import load_dqn_model, get_device


class HybridSolver:
    """Hybrid solver using DQN warm-start and ALNS refinement."""
    
    def __init__(self, instance: VRPTWInstance, time_limit: float = 15.0):
        self.instance = instance
        self.time_limit = time_limit
    
    def solve(self) -> Solution:
        """Solve using DQN construction + ALNS refinement."""
        try:
            model = load_dqn_model()
            initial_routes = self._dqn_construct(model)
        except Exception:
            initial_routes = None
        
        alns = ALNSSolver(self.instance, self.time_limit)
        return alns.solve(initial_routes)
    
    def _dqn_construct(self, model) -> List[List[int]]:
        """Construct initial solution using DQN."""
        inst = self.instance
        device = get_device()
        n = inst.n_customers
        
        # Normalize features
        features = np.zeros((n, 5), dtype=np.float32)
        for i, cust in enumerate(inst.customers):
            features[i, 0] = cust.x / inst.max_coord
            features[i, 1] = cust.y / inst.max_coord
            features[i, 2] = cust.demand / max(inst.max_demand, 1.0)
            features[i, 3] = cust.ready_time / inst.max_time
            features[i, 4] = cust.due_date / inst.max_time
        
        routes = []
        unvisited = set(range(n))
        current_route = []
        current_node = 0  # Depot
        current_time = 0.0
        current_load = 0.0
        
        while unvisited:
            # Build mask for infeasible actions
            mask = np.ones(n, dtype=bool)
            
            for i in unvisited:
                cust = inst.customers[i]
                travel = inst.distance(current_node, i + 1)
                arrival = max(current_time + travel, cust.ready_time)
                
                if arrival <= cust.due_date and current_load + cust.demand <= inst.vehicle_capacity:
                    mask[i] = False
            
            if mask.all():
                # No feasible customer, start new route
                if current_route:
                    routes.append(current_route)
                current_route = []
                current_node = 0
                current_time = 0.0
                current_load = 0.0
                continue
            
            # Prepare state
            state = np.array([
                inst.depot.x / inst.max_coord if current_node == 0 else inst.customers[current_node - 1].x / inst.max_coord,
                current_load / inst.vehicle_capacity,
                current_time / inst.max_time
            ], dtype=np.float32)
            
            # Get action from DQN
            with torch.no_grad():
                f_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                m_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device)
                
                q_values = model(f_t, s_t, m_t)
                action = q_values.argmax(dim=-1).item()
            
            if mask[action]:
                valid = np.where(~mask)[0]
                action = valid[0] if len(valid) > 0 else None
            
            if action is None:
                break
            
            # Apply action
            cust = inst.customers[action]
            node_id = action + 1
            travel = inst.distance(current_node, node_id)
            arrival = max(current_time + travel, cust.ready_time)
            
            current_time = arrival + cust.service_time
            current_load += cust.demand
            current_node = node_id
            current_route.append(node_id)
            unvisited.remove(action)
        
        if current_route:
            routes.append(current_route)
        
        return routes
