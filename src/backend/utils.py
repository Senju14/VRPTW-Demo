import numpy as np

class VRPTWInstance:
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        self.name = lines[0].strip()
        self.capacity = float(lines[4].split()[1])
        
        data = []
        for line in lines[9:]:
            if line.strip():
                data.append(list(map(float, line.split())))
        
        self.data = np.array(data)
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
        return self.dist_matrix[i, j]

class Solution:
    def __init__(self, routes, instance):
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
    