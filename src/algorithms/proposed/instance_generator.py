"""
Solomon-style Instance Generator for DQN-ALNS Training.

Generates synthetic VRPTW instances matching the characteristics of
Solomon benchmark datasets (R, C, RC distributions).

Reference: Solomon (1987) benchmark instances.
"""

import random
import math
import numpy as np
from typing import Tuple, List
from ...schemas import VRPTWInstance, Customer


def generate_solomon_style_instance(
    n_customers: int = 100,
    distribution: str = "RC",
    tw_tightness: float = 0.5,
    capacity: float = 200.0,
    service_time_range: Tuple[float, float] = (10.0, 20.0),
    coord_range: Tuple[float, float] = (0.0, 100.0),
    seed: int = None
) -> VRPTWInstance:
    """
    Generate a synthetic VRPTW instance in Solomon style.
    
    Args:
        n_customers: Number of customers (excl. depot)
        distribution: "R" (random), "C" (clustered), "RC" (mixed)
        tw_tightness: 0.0 (wide windows) to 1.0 (tight windows)
        capacity: Vehicle capacity
        service_time_range: (min, max) service time
        coord_range: (min, max) coordinate values
        seed: Random seed for reproducibility
    
    Returns:
        VRPTWInstance ready for ALNS solving
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    min_coord, max_coord = coord_range
    center = (max_coord + min_coord) / 2
    
    # Generate customer locations based on distribution type
    if distribution == "R":
        positions = _generate_random_positions(n_customers, coord_range)
    elif distribution == "C":
        positions = _generate_clustered_positions(n_customers, coord_range)
    elif distribution == "RC":
        # Half random, half clustered
        n_random = n_customers // 2
        n_clustered = n_customers - n_random
        pos_random = _generate_random_positions(n_random, coord_range)
        pos_clustered = _generate_clustered_positions(n_clustered, coord_range)
        positions = pos_random + pos_clustered
        random.shuffle(positions)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Depot at center
    depot = Customer(
        id=0,
        x=center,
        y=center,
        demand=0,
        ready_time=0,
        due_date=max_coord * 5,  # Large horizon
        service_time=0
    )
    
    # Generate customers with time windows
    customers = []
    horizon = depot.due_date
    
    for i, (x, y) in enumerate(positions):
        # Demand: proportional to capacity
        demand = random.uniform(5, capacity * 0.3)
        
        # Service time
        service_time = random.uniform(*service_time_range)
        
        # Time window based on distance from depot and tightness
        dist_from_depot = math.sqrt((x - center)**2 + (y - center)**2)
        travel_time = dist_from_depot  # Assume unit speed
        
        # Earliest arrival based on travel time
        ready_time = travel_time + random.uniform(0, horizon * 0.3)
        
        # Window width inversely proportional to tightness
        max_width = horizon * 0.4 * (1 - tw_tightness * 0.8)
        window_width = random.uniform(service_time + 10, max_width)
        
        due_date = min(ready_time + window_width, horizon)
        
        # Ensure feasibility
        ready_time = min(ready_time, due_date - service_time)
        
        customers.append(Customer(
            id=i + 1,
            x=x,
            y=y,
            demand=demand,
            ready_time=ready_time,
            due_date=due_date,
            service_time=service_time
        ))
    
    # Estimate number of vehicles needed
    total_demand = sum(c.demand for c in customers)
    num_vehicles = int(math.ceil(total_demand / capacity)) + 5
    
    return VRPTWInstance(
        name=f"synthetic_{distribution}_{n_customers}_{seed}",
        depot=depot,
        customers=customers,
        vehicle_capacity=capacity,
        num_vehicles=num_vehicles
    )


def _generate_random_positions(
    n: int, 
    coord_range: Tuple[float, float]
) -> List[Tuple[float, float]]:
    """Generate uniformly random positions."""
    min_c, max_c = coord_range
    return [(random.uniform(min_c, max_c), random.uniform(min_c, max_c)) 
            for _ in range(n)]


def _generate_clustered_positions(
    n: int, 
    coord_range: Tuple[float, float],
    n_clusters: int = None
) -> List[Tuple[float, float]]:
    """Generate clustered positions around random centers."""
    min_c, max_c = coord_range
    range_size = max_c - min_c
    
    if n_clusters is None:
        n_clusters = max(3, n // 10)
    
    # Generate cluster centers
    centers = [(random.uniform(min_c + range_size*0.2, max_c - range_size*0.2),
                random.uniform(min_c + range_size*0.2, max_c - range_size*0.2))
               for _ in range(n_clusters)]
    
    # Generate points around centers
    positions = []
    cluster_std = range_size / (2 * n_clusters)
    
    for i in range(n):
        cx, cy = random.choice(centers)
        x = np.clip(np.random.normal(cx, cluster_std), min_c, max_c)
        y = np.clip(np.random.normal(cy, cluster_std), min_c, max_c)
        positions.append((x, y))
    
    return positions


def generate_training_batch(
    batch_size: int = 32,
    n_customers: int = 100,
    distributions: List[str] = ["R", "C", "RC"],
    tw_tightness_range: Tuple[float, float] = (0.3, 0.7),
    seed_base: int = None
) -> List[VRPTWInstance]:
    """
    Generate a batch of diverse training instances.
    
    Args:
        batch_size: Number of instances to generate
        n_customers: Number of customers per instance
        distributions: List of distribution types to sample from
        tw_tightness_range: Range of time window tightness
        seed_base: Base seed for reproducibility
    
    Returns:
        List of VRPTWInstance objects
    """
    instances = []
    
    for i in range(batch_size):
        dist = random.choice(distributions)
        tw = random.uniform(*tw_tightness_range)
        seed = (seed_base + i) if seed_base is not None else None
        
        instance = generate_solomon_style_instance(
            n_customers=n_customers,
            distribution=dist,
            tw_tightness=tw,
            seed=seed
        )
        instances.append(instance)
    
    return instances


def generate_scaling_test_instances(
    sizes: List[int] = [25, 50, 75, 100],
    seeds_per_size: int = 5
) -> List[VRPTWInstance]:
    """
    Generate test instances for generalization evaluation.
    
    Tests if model trained on one size can generalize to others.
    """
    instances = []
    
    for size in sizes:
        for seed in range(seeds_per_size):
            for dist in ["R", "C", "RC"]:
                instance = generate_solomon_style_instance(
                    n_customers=size,
                    distribution=dist,
                    tw_tightness=0.5,
                    seed=size * 1000 + seed * 10 + {"R": 1, "C": 2, "RC": 3}[dist]
                )
                instances.append(instance)
    
    return instances
