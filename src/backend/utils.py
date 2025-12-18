import numpy as np
from typing import List, Tuple

class VRPTWInstance:
    def __init__(self, instance_dict):
        self.name = instance_dict['name']
        self.capacity = instance_dict['capacity']
        self.data = np.array([
            [0, instance_dict['depot']['lat'], instance_dict['depot']['lng'], 0, 
             instance_dict['depot']['ready_time'], instance_dict['depot']['due_time'], 
             instance_dict['depot']['service_time']]
        ] + [
            [c['id'], c['lat'], c['lng'], c['demand'], 
             c['ready_time'], c['due_time'], c['service_time']]
            for c in instance_dict['customers']
        ])
        
        self.n = len(self.data) - 1
        self.coords = self.data[:, 1:3]
        self.demands = self.data[:, 3]
        self.ready_times = self.data[:, 4]
        self.due_times = self.data[:, 5]
        self.service_times = self.data[:, 6]

        self.dist_matrix = self._compute_distances()

    def _compute_distances(self):
        coords = self.coords
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))

    def distance(self, i, j):
        return self.dist_matrix[int(i), int(j)]

class Solution:
    def __init__(self, routes: List[List[int]], instance: VRPTWInstance):
        self.routes = routes
        self.instance = instance
        self.cost = self._compute_cost()
        self.feasible = self._check_feasibility()

    def _compute_cost(self):
        total = 0
        for route in self.routes:
            if len(route) == 0:
                continue
            total += self.instance.distance(0, route[0])
            for i in range(len(route) - 1):
                total += self.instance.distance(route[i], route[i+1])
            total += self.instance.distance(route[-1], 0)
        return total

    def _check_feasibility(self):
        inst = self.instance
        for route in self.routes:
            if sum(inst.demands[i] for i in route) > inst.capacity:
                return False

            time = 0
            current = 0
            for node in route:
                time += inst.distance(current, node)
                time = max(time, inst.ready_times[node])
                if time > inst.due_times[node]:
                    return False
                time += inst.service_times[node]
                current = node
        return True

    def copy(self):
        return Solution([r[:] for r in self.routes], self.instance)

    def __repr__(self):
        return f"Solution(routes={len(self.routes)}, cost={self.cost:.2f}, feasible={self.feasible})"

def compute_route_metrics(route: List[int], instance: VRPTWInstance) -> Tuple[float, float, bool]:
    if not route:
        return 0, 0, True

    load = sum(instance.demands[i] for i in route)

    distance = instance.distance(0, route[0])
    for i in range(len(route) - 1):
        distance += instance.distance(route[i], route[i+1])
    distance += instance.distance(route[-1], 0)

    time = 0
    current = 0
    feasible = load <= instance.capacity

    for node in route:
        time += instance.distance(current, node)
        time = max(time, instance.ready_times[node])
        if time > instance.due_times[node]:
            feasible = False
        time += instance.service_times[node]
        current = node

    return distance, load, feasible
