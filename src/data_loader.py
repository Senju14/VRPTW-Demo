"""Solomon benchmark instance loader."""

from .schemas import Customer, VRPTWInstance

def parse_solomon_file(filepath: str) -> VRPTWInstance:
    """Parse Solomon format benchmark file and return a VRPTWInstance."""
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
    
    return VRPTWInstance(
        name=name,
        depot=depot,
        customers=customers,
        vehicle_capacity=capacity,
        num_vehicles=num_vehicles
    )
    
# Test loading a Solomon instance - python -m src.data_loader
if __name__ == "__main__":

    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(base_dir, "data", "Solomon", "RC201.txt")
    
    if os.path.exists(test_file):
        instance = parse_solomon_file(test_file)
        print(f"Loaded instance: {instance.name}")
        print(f"Customers: {len(instance.customers)}")
        print(f"Capacity: {instance.vehicle_capacity}")
    else:
        print(f"Test file not found at: {test_file}")
