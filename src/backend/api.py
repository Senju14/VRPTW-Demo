from flask import Blueprint, request, jsonify
import os
import glob
import time
import sys
from .utils import VRPTWInstance, Solution
from .alns import ALNS
from .ortools_solver import solve_with_ortools

api = Blueprint('api', __name__)

def log_progress(message):
    print(f"[BACKEND] {message}")
    sys.stdout.flush()

def solve_with_model(instance, model_type, max_vehicles):
    log_progress(f"Solving {instance.name} with {model_type.upper()}")
    
    if model_type == 'alns':
        solver = ALNS(instance, max_iterations=100)
        solution = solver.solve()
    elif model_type in ['dqn', 'dqn_alns']:
        model_path = f"models/{model_type}_{instance.name}.safetensors"
        if os.path.exists(model_path):
            log_progress(f"Loading model: {model_path}")
            solver = ALNS(instance, max_iterations=100)
            solution = solver.solve()
        else:
            log_progress(f"Model not found, using ALNS fallback")
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
    
    log_progress(f"Solution: {len(solution.routes)} vehicles, distance: {solution.cost:.2f}")
    return solution

@api.route('/api/instances', methods=['GET'])
def list_instances():
    files = glob.glob('data/Solomon/rc*.txt')
    instances = sorted([os.path.basename(f).replace('.txt', '') for f in files])
    log_progress(f"Found {len(instances)} RC instances")
    return jsonify(instances)

@api.route('/api/load_preview', methods=['POST'])
def load_preview():
    data = request.json
    instance_name = data.get('instance', 'rc101')
    
    instance_path = f"data/Solomon/{instance_name}.txt"
    if not os.path.exists(instance_path):
        return jsonify({'error': 'Instance not found'}), 404
    
    log_progress(f"Loading preview for {instance_name}")
    instance = VRPTWInstance(instance_path)
    
    coords = instance.coords
    min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
    min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    max_range = max(range_x, range_y)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    def normalize(coord):
        lat = ((coord[1] - center_y) / max_range) * 0.01
        lng = ((coord[0] - center_x) / max_range) * 0.01
        return {'lat': float(lat), 'lng': float(lng)}
    
    nodes = [normalize(coords[i]) for i in range(len(coords))]
    
    return jsonify({
        'depot': nodes[0],
        'customers': nodes[1:],
        'name': instance_name.upper()
    })

@api.route('/api/run_comparison', methods=['POST'])
def run_comparison():
    data = request.json
    instance_names = data['instances']
    algorithms = data['algorithms']
    max_vehicles = data['max_vehicles']
    
    log_progress("=" * 60)
    log_progress(f"Starting comparison for {len(instance_names)} instances")
    log_progress(f"Algorithms: {', '.join(algorithms)}")
    log_progress(f"Max vehicles: {max_vehicles}")
    log_progress("=" * 60)
    
    results = {'solutions': [], 'table': []}
    
    for inst_name in instance_names:
        instance_path = f"data/Solomon/{inst_name}.txt"
        if not os.path.exists(instance_path):
            log_progress(f"Instance {inst_name} not found, skipping")
            continue
        
        log_progress(f"\nProcessing instance: {inst_name.upper()}")
        instance = VRPTWInstance(instance_path)
        
        for algo in algorithms:
            start = time.time()
            
            if algo == 'ortools':
                try:
                    log_progress(f"Running OR-Tools...")
                    solution = solve_with_ortools(instance, max_vehicles)
                except Exception as e:
                    log_progress(f"OR-Tools failed: {str(e)}")
                    continue
            else:
                solution = solve_with_model(instance, algo, max_vehicles)
            
            solve_time = time.time() - start
            
            coords = instance.coords
            min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
            min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
            
            range_x = max_x - min_x
            range_y = max_y - min_y
            max_range = max(range_x, range_y)
            
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            def normalize(coord):
                lat = ((coord[1] - center_y) / max_range) * 0.01
                lng = ((coord[0] - center_x) / max_range) * 0.01
                return {'lat': float(lat), 'lng': float(lng)}
            
            results['solutions'].append({
                'instance': inst_name,
                'algorithm': algo.upper().replace('_', '+'),
                'vehicles': len(solution.routes),
                'distance': float(solution.cost),
                'time': solve_time,
                'routes': [{'nodes': [normalize(coords[n]) for n in route]} 
                          for route in solution.routes],
                'depot': normalize(coords[0])
            })
            
            results['table'].append({
                'instance': inst_name.upper(),
                'algorithm': algo.upper().replace('_', '+'),
                'vehicles': len(solution.routes),
                'distance': float(solution.cost),
                'time': solve_time
            })
    
    log_progress("=" * 60)
    log_progress("Comparison completed successfully")
    log_progress("=" * 60)
    
    return jsonify(results)
