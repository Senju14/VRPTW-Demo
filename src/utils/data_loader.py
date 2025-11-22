"""
Data loader for VRPTW instances.
"""

import os
from typing import List, Tuple, Dict, Any


class Customer:
    def __init__(self, id: int, x: float, y: float, demand: int, ready_time: int, due_date: int, service_time: int):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time


def load_instance(instance_path: str) -> Tuple[Customer, List[Customer], int, int]:
    """
    Load a VRPTW instance from file.

    Args:
        instance_path: Path to the instance file (with or without extension).

    Returns:
        depot: Customer object for depot.
        customers: List of Customer objects.
        vehicle_capacity: Capacity of vehicles.
        num_vehicles: Number of vehicles.
    """
    # Handle path with extension already included
    file_path = instance_path
    if not os.path.exists(file_path):
        # Try adding .TXT extension
        if not instance_path.endswith(('.txt', '.TXT')):
            file_path = instance_path + ".TXT"
            if not os.path.exists(file_path):
                file_path = instance_path + ".txt"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Instance file not found: {instance_path}")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip header
    idx = 0
    while idx < len(lines) and not lines[idx].strip().startswith("VEHICLE"):
        idx += 1
    idx += 1  # Skip "NUMBER     CAPACITY" header
    
    # Find the line with actual numbers
    while idx < len(lines):
        line = lines[idx].strip()
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            num_vehicles = int(parts[0])
            vehicle_capacity = int(parts[1])
            break
        idx += 1

    # Skip to CUSTOMER
    while idx < len(lines) and not lines[idx].strip().startswith("CUSTOMER"):
        idx += 1
    idx += 1  # Skip "CUSTOMER" line
    
    # Skip header lines until we find actual data
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or line.startswith("CUST NO.") or line.startswith("---"):
            idx += 1
            continue
        # Found data line
        break

    customers = []
    for line in lines[idx:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            demand = int(parts[3])
            ready = int(parts[4])
            due = int(parts[5])
            service = int(parts[6])
            customers.append(Customer(id, x, y, demand, ready, due, service))
        except (ValueError, IndexError):
            # Skip invalid lines
            continue

    depot = customers[0]
    customers = customers[1:]

    return depot, customers, vehicle_capacity, num_vehicles


def get_available_instances() -> List[Dict[str, str]]:
    """
    Get list of available instances from manifest.
    """
    import json
    manifest_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'file_manifest.json')
    with open(manifest_path, 'r') as f:
        return json.load(f)