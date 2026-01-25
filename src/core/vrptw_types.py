"""VRPTW data types and structures."""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


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


def parse_solomon_file(filepath: str) -> dict:
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
    
    return {
        'name': name,
        'depot': depot,
        'customers': customers,
        'capacity': capacity,
        'num_vehicles': num_vehicles,
        'raw_nodes': nodes
    }
