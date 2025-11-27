"""
Flask backend for VRPTW Demo.
"""

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import os
from src.utils.data_loader import load_instance, get_available_instances
from src.utils.ortools_solver import solve_vrptw
from src.utils.visualization import create_map, get_map_html
from src.utils.dqn_solver import solve_dqn_vrptw

# DQN dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
import numpy as np

app = Flask(__name__, template_folder='../frontend', static_folder='../frontend/static')
CORS(app)  # Enable CORS for frontend

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data/<path:filename>')
def serve_data_file(filename):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    return send_from_directory(data_dir, filename)

@app.route('/api/instances', methods=['GET'])
def get_instances():
    try:
        instances = []
        
        # Load Solomon instances
        solomon_path = 'data/Solomon/'
        if os.path.exists(solomon_path):
            for filename in os.listdir(solomon_path):
                if filename.endswith('.txt'):
                    name = filename.replace('.txt', '').upper()
                    # Check if corresponding model exists
                    model_file = f'models/dqn_{name.lower()}.safetensor'
                    has_model = os.path.exists(model_file)
                    instances.append({
                        'name': name,
                        'group': 'Solomon Benchmark',
                        'path': f'Solomon/{filename}',
                        'model_path': model_file if has_model else None,
                        'has_model': has_model
                    })
        
        # Load Gehring-Homberger instances - all customer sizes
        gehring_base = 'data/Gehring_Homberger/'
        gehring_folders = {
            'homberger_200_customer_instances': '200 customers',
            'homberger_400_customer_instances': '400 customers', 
            'homberger_600_customer_instances': '600 customers',
            'homberger_800_customer_instances': '800 customers',
            'homberger_1000_customer_instances': '1000 customers'
        }
        
        for folder, size_label in gehring_folders.items():
            folder_path = gehring_base + folder + '/'
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith('.TXT'):
                        name = filename.replace('.TXT', '')
                        # Check for corresponding model
                        customer_count = folder.split('_')[1]  # Extract customer count
                        model_file = f'models/dqn_{customer_count}_{name}.safetensor'
                        has_model = os.path.exists(model_file)
                        
                        instances.append({
                            'name': name,
                            'group': f'Gehring-Homberger ({size_label})',
                            'path': f'Gehring_Homberger/{folder}/{filename}',
                            'model_path': model_file if has_model else None,
                            'has_model': has_model
                        })
        
        # Sort instances by group then by name
        instances.sort(key=lambda x: (x['group'], x['name']))
        
        return jsonify(instances)
    
    except Exception as e:
        print(f"Error loading instances: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/load_instance', methods=['POST'])
def load_instance_api():
    data = request.json
    instance_path = data['instance_path']
    print(f"Loading instance: {instance_path}")  # Debug log
    try:
        depot, customers, capacity, num_vehicles = load_instance(instance_path)
        # Return basic info
        instance_data = {
            'depot': {'id': depot.id, 'x': depot.x, 'y': depot.y, 'demand': depot.demand},
            'customers': [{'id': c.id, 'x': c.x, 'y': c.y, 'demand': c.demand, 'ready_time': c.ready_time, 'due_date': c.due_date, 'service_time': c.service_time} for c in customers],
            'capacity': capacity,
            'num_vehicles': num_vehicles
        }
        return jsonify(instance_data)
    except Exception as e:
        print(f"Error loading instance {instance_path}: {str(e)}")  # Debug log
        return jsonify({'error': str(e)}), 400

@app.route('/api/solve_demo', methods=['POST'])
def solve_demo():
    data = request.json
    instance_path = data['instance_path']
    instance_name = data.get('instance_name', '')
    num_vehicles = data.get('num_vehicles', 5)
    
    print(f"\n{'='*60}")
    print(f"[SOLVE_DEMO] Request received")
    print(f"[SOLVE_DEMO] Instance: {instance_name}")
    print(f"[SOLVE_DEMO] Path: {instance_path}")
    print(f"[SOLVE_DEMO] Vehicles: {num_vehicles}")
    print(f"{'='*60}\n")
    
    try:
        # Load instance
        print(f"[SOLVE_DEMO] Loading instance data...")
        depot, customers, capacity, _ = load_instance(instance_path)
        print(f"[SOLVE_DEMO] Loaded: {len(customers)} customers, capacity: {capacity}")
        
        # Determine correct model path based on instance
        model_path = None
        if 'Solomon' in instance_path:
            # Solomon instances: dqn_c101.safetensor format
            instance_code = instance_name.lower()
            model_path = f'models/dqn_{instance_code}.safetensor'
        elif 'Gehring_Homberger' in instance_path:
            # Gehring-Homberger instances: dqn_200_C1_2_1.safetensor format
            if 'homberger_200_customer' in instance_path:
                model_path = f'models/dqn_200_{instance_name}.safetensor'
            elif 'homberger_1000_customer' in instance_path:
                model_path = f'models/dqn_1000_{instance_name}.safetensor'
            # Add other customer sizes as needed
        
        # Check if model exists
        model_exists = model_path and os.path.exists(model_path)
        print(f"[SOLVE_DEMO] Model path: {model_path}")
        print(f"[SOLVE_DEMO] Model exists: {model_exists}")
        
        # Choose solver based on model availability
        if model_exists:
            print(f"[SOLVE_DEMO] Using DQN Learned Model")
            results = solve_dqn_vrptw(depot, customers, capacity, num_vehicles, model_path)
        else:
            print(f"[SOLVE_DEMO] Using OR-Tools Baseline")
            results = solve_vrptw(depot, customers, capacity, num_vehicles)
        
        print(f"[SOLVE_DEMO] Solver completed successfully")
        
        # Add solver info to results
        results['solver_used'] = f'DQN Model: {model_path}' if model_exists else 'OR-Tools Baseline'
        results['using_learned_model'] = model_exists
        
        # Calculate additional metrics
        total_customers = len(customers)
        customers_served = sum(len(route) - 1 for route in results['routes'] if route)  # -1 for depot
        coverage = (customers_served / total_customers * 100) if total_customers > 0 else 0
        vehicles_used = sum(1 for route in results['routes'] if len(route) > 1)
        avg_distance = results['total_distance'] / vehicles_used if vehicles_used > 0 else 0
        
        results.update({
            'customers_served': customers_served,
            'coverage': f"{coverage:.1f}%",
            'vehicles_used': vehicles_used,
            'avg_distance': avg_distance
        })
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/solve', methods=['POST'])
def solve():
    data = request.json
    instance_path = data['instance_path']
    num_vehicles = data['num_vehicles']
    try:
        depot, customers, capacity, _ = load_instance(instance_path)
        results = solve_vrptw(depot, customers, capacity, num_vehicles)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/map', methods=['POST'])
def get_map():
    data = request.json
    instance_path = data['instance_path']
    routes = data.get('routes', None)
    try:
        depot, customers, capacity, _ = load_instance(instance_path)
        m = create_map(depot, customers, routes)
        map_html = get_map_html(m)
        return jsonify({'map_html': map_html})
    except Exception as e:
        return jsonify({'error': str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)