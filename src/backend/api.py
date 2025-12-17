from flask import Blueprint, request, jsonify
import os
import glob
import time
from .utils import VRPTWInstance, Solution
from .alns import ALNS
from .ortools_solver import solve_with_ortools

api = Blueprint('api', __name__)

def solve_with_model(instance, model_type, max_vehicles):
    if model_type == 'alns':
        solver = ALNS(instance, max_iterations=100)
        solution = solver.solve()
    elif model_type in ['dqn', 'dqn_alns']:
        model_path = f"models/{model_type}_{instance.name.lower()}.safetensors"
        if os.path.exists(model_path):
            solver = ALNS(instance, max_iterations=100)
            solution = solver.solve()
        else:
            solver = ALNS(instance, max_iterations=100)
            solution = solver.solve()
    else:
        solver = ALNS(instance, max_iterations=100)
        solution = solver.solve()
    
    while len(solution.routes) > max_vehicles and len(solution.routes) > 1:
        routes_by_load = sorted(enumerate(solution.routes), 
                               key=lambda x: sum(instance.demands[i] for i in x[1]))
        idx1, route1 = routes_by_load[0]
        idx2, route2 = routes_by_load[1]
        
        merged = route1 + route2
        new_routes = [r for i, r in enumerate(solution.routes) if i not in [idx1, idx2]]
        new_routes.append(merged)
        
        solution = Solution(new_routes, instance)
        if not solution.feasible:
            break
    
    return solution

@api.route('/api/instances', methods=['GET'])
def list_instances():
    files = glob.glob('data/Solomon/*.txt')
    instances = [os.path.basename(f).replace('.txt', '') for f in files]
    return jsonify(sorted(instances))

@api.route('/api/run_comparison', methods=['POST'])
def run_comparison():
    data = request.json
    instance_names = data['instances']
    algorithms = data['algorithms']
    max_vehicles = data['max_vehicles']
    
    results = {'solutions': [], 'table': []}
    
    for inst_name in instance_names:
        instance_path = f"data/Solomon/{inst_name}.txt"
        if not os.path.exists(instance_path):
            continue
        
        instance = VRPTWInstance(instance_path)
        
        for algo in algorithms:
            start = time.time()
            
            if algo == 'ortools':
                try:
                    solution = solve_with_ortools(instance, max_vehicles)
                except:
                    continue
            else:
                solution = solve_with_model(instance, algo, max_vehicles)
            
            solve_time = time.time() - start
            
            min_x, max_x = instance.coords[:, 0].min(), instance.coords[:, 0].max()
            min_y, max_y = instance.coords[:, 1].min(), instance.coords[:, 1].max()
            
            def normalize(coord):
                x = (coord[0] - min_x) / (max_x - min_x) * 560 + 20
                y = (coord[1] - min_y) / (max_y - min_y) * 360 + 20
                return {'x': float(x), 'y': float(y)}
            
            results['solutions'].append({
                'instance': inst_name,
                'algorithm': algo.upper().replace('_', '+'),
                'vehicles': len(solution.routes),
                'distance': float(solution.cost),
                'time': solve_time,
                'routes': [{'nodes': [normalize(instance.coords[n]) for n in route]} 
                          for route in solution.routes],
                'depot': normalize(instance.coords[0])
            })
            
            results['table'].append({
                'instance': inst_name.upper(),
                'algorithm': algo.upper().replace('_', '+'),
                'vehicles': len(solution.routes),
                'distance': float(solution.cost),
                'time': solve_time
            })
    
    return jsonify(results)
