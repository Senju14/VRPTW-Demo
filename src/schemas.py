"""
Data types and Pydantic schemas for VRPTW.
Combined Dataclasses for algorithms and Pydantic models for API.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pydantic import BaseModel


# =============================================================================
# ALGORITHM CORE TYPES (Dataclasses)
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
    
    def __post_init__(self):
        all_nodes = [self.depot] + self.customers
        self.max_coord = max(max(c.x for c in all_nodes), max(c.y for c in all_nodes))
        self.max_time = max(c.due_date for c in all_nodes)
        self.max_demand = max((c.demand for c in self.customers), default=1.0)
        self._build_distance_matrix()
    
    def _build_distance_matrix(self):
        """Precompute distance matrix for all nodes."""
        n = len(self.customers) + 1
        self.dist_matrix = np.zeros((n, n), dtype=np.float32)
        all_nodes = [self.depot] + self.customers
        
        for i, n1 in enumerate(all_nodes):
            for j, n2 in enumerate(all_nodes):
                self.dist_matrix[i, j] = math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
    
    def distance(self, i: int, j: int) -> float:
        """Get distance between nodes i and j (0 = depot)."""
        return float(self.dist_matrix[i, j])
    
    @property
    def n_customers(self) -> int:
        return len(self.customers)


@dataclass
class Solution:
    """A solution to a VRPTW instance."""
    routes: List[List[int]]  # Each route is list of customer indices (1-indexed)
    instance: VRPTWInstance
    
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
        all_nodes = [inst.depot] + inst.customers
        
        for route in self.routes:
            load = sum(all_nodes[i].demand for i in route)
            if load > inst.vehicle_capacity:
                return False
            
            time = 0.0
            prev = 0
            for node_id in route:
                node = all_nodes[node_id]
                time += inst.distance(prev, node_id)
                time = max(time, node.ready_time)
                if time > node.due_date:
                    return False
                time += node.service_time
                prev = node_id
        
        return True
    
    @property
    def num_vehicles(self) -> int:
        return len([r for r in self.routes if r])
    
    def copy(self) -> 'Solution':
        return Solution([r[:] for r in self.routes], self.instance)


# =============================================================================
# API DATA MODELS (Pydantic)
# =============================================================================

class NodeData(BaseModel):
    """Information for a node (depot or customer)"""
    id: int
    lat: float
    lng: float
    demand: float = 0
    ready_time: float = 0
    due_time: float = 0
    service_time: float = 0
    address: Optional[str] = None
    arrival_time: Optional[float] = None
    start_service: Optional[float] = None
    end_service: Optional[float] = None
    wait_time: Optional[float] = None


class RouteData(BaseModel):
    """Route containing a list of nodes"""
    nodes: List[NodeData]
    vehicle_id: Optional[int] = None
    distance: float = 0
    duration: float = 0
    geometry: Optional[List[Tuple[float, float]]] = None


class SolveRequest(BaseModel):
    """Solve request for benchmark or custom problem."""
    instance: Optional[str] = None
    algorithms: List[str]
    max_vehicles: Optional[int] = None
    time_limit: float = 15.0
    # For custom problem
    customers: Optional[List[NodeData]] = None
    depot: Optional[NodeData] = None


class SolutionResult(BaseModel):
    """Algorithm result summary"""
    algorithm: str
    vehicles: int
    distance: float
    time: float
    routes: Optional[List[RouteData]] = None
    depot: Optional[NodeData] = None
    error: Optional[str] = None


class SolveResponse(BaseModel):
    """Response containing multiple solutions"""
    solutions: List[SolutionResult]
