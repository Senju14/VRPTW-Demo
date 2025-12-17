import os
import json
import csv
import io
import traceback
from datetime import datetime
from flask import Flask, jsonify, request, render_template, send_from_directory, Response
from flask_cors import CORS

from src.backend.data_loader import load_instance
from src.backend.solver import solve_vrptw, solve_alns_vrptw, solve_dqn_only_vrptw, solve_dqn_alns_vrptw
from src.backend.visualization import create_map, get_map_html

app = Flask(__name__, template_folder='../frontend', static_folder='../frontend/static')
CORS(app)

def get_model_path(instance_path, instance_name):
    if 'Solomon' in instance_path:
        return f'models/dqn_{instance_name.lower()}.safetensor'
    elif 'Gehring_Homberger' in instance_path:
        if 'homberger_200_customer' in instance_path:
            return f'models/dqn_200_{instance_name}.safetensor'
        elif 'homberger_1000_customer' in instance_path:
            return f'models/dqn_1000_{instance_name}.safetensor'
    return None

def calculate_metrics(routes, total_customers, total_distance, execution_time=0):
    customers_served = sum(len(r) - 1 for r in routes if r)
    vehicles_used = sum(1 for r in routes if len(r) > 1)
    coverage = (customers_served / total_customers * 100) if total_customers > 0 else 0
    avg_dist = total_distance / vehicles_used if vehicles_used > 0 else 0
    return {
        'customers_served': customers_served,
        'vehicles_used': vehicles_used,
        'coverage': round(coverage, 2),
        'avg_distance': round(avg_dist, 2),
        'total_distance': round(total_distance, 2),
        'execution_time': round(execution_time, 2)
    }

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
        # Sử dụng đường dẫn tuyệt đối giống như serve_data_file
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        solomon_path = os.path.join(base_dir, 'data', 'Solomon')
        
        if os.path.exists(solomon_path):
            for f in os.listdir(solomon_path):
                # Chỉ lấy các file RC (rc101, rc102, rc201, rc202, ...)
                if f.endswith('.txt') and f.lower().startswith('rc'):
                    name = f.replace('.txt', '').upper()
                    model_path = os.path.join(base_dir, 'models', f'dqn_{name.lower()}.safetensor')
                    instances.append({
                        'name': name, 'group': 'Solomon Benchmark',
                        'path': f'Solomon/{f}', 
                        'model_path': model_path if os.path.exists(model_path) else None,
                        'has_model': os.path.exists(model_path)
                    })

        # Bỏ qua Gehring-Homberger vì chỉ cần RC instances
        
        return jsonify(sorted(instances, key=lambda x: (x['group'], x['name'])))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load_instance', methods=['POST'])
def load_instance_api():
    try:
        d, c, cap, num_v = load_instance(request.json['instance_path'])
        return jsonify({
            'depot': {'id': d.id, 'x': d.x, 'y': d.y, 'demand': d.demand},
            'customers': [{'id': x.id, 'x': x.x, 'y': x.y, 'demand': x.demand, 
                           'ready_time': x.ready_time, 'due_date': x.due_date, 'service_time': x.service_time} for x in c],
            'capacity': cap, 'num_vehicles': num_v
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/solve_demo', methods=['POST'])
def solve_demo():
    try:
        data = request.json
        d, c, cap, _ = load_instance(data['instance_path'])
        mode = data.get('solver_mode', 'auto')
        model_path = get_model_path(data['instance_path'], data.get('instance_name', ''))
        
        if mode == 'auto':
            mode = 'dqn_alns' if model_path and os.path.exists(model_path) else 'ortools'

        if mode == 'alns':
            res = solve_alns_vrptw(d, c, cap, data.get('num_vehicles', 5))
            name = 'ALNS (Pure)'
        elif mode == 'dqn':
            res = solve_dqn_only_vrptw(d, c, cap, data.get('num_vehicles', 5), model_path)
            name = 'DQN (Pure)'
        elif mode == 'dqn_alns':
            res = solve_dqn_alns_vrptw(d, c, cap, data.get('num_vehicles', 5), model_path)
            name = 'ALNS + DQN'
        else:
            res = solve_vrptw(d, c, cap, data.get('num_vehicles', 5))
            name = 'OR-Tools'

        metrics = calculate_metrics(res['routes'], len(c), res['total_distance'])
        res.update(metrics)
        res.update({'solver_used': name, 'solver_mode': mode, 'coverage': f"{metrics['coverage']}%"})
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/api/compare_solvers', methods=['POST'])
def compare_solvers():
    try:
        data = request.json
        d, c, cap, _ = load_instance(data['instance_path'])
        model_path = get_model_path(data['instance_path'], data.get('instance_name', ''))
        results = []

        for mode in data.get('solvers', []):
            res_entry = {'solver': mode, 'success': False}
            try:
                if mode == 'alns':
                    sol = solve_alns_vrptw(d, c, cap, data.get('num_vehicles', 5))
                    s_name = 'ALNS'
                elif mode == 'dqn':
                    sol = solve_dqn_only_vrptw(d, c, cap, data.get('num_vehicles', 5), model_path)
                    s_name = 'DQN'
                elif mode == 'dqn_alns':
                    sol = solve_dqn_alns_vrptw(d, c, cap, data.get('num_vehicles', 5), model_path)
                    s_name = 'ALNS+DQN'
                else:
                    sol = solve_vrptw(d, c, cap, data.get('num_vehicles', 5))
                    s_name = 'OR-Tools'
                
                metrics = calculate_metrics(sol['routes'], len(c), sol['total_distance'], sol.get('execution_time', 0))
                res_entry.update({'success': True, 'solver_name': s_name, 'routes': sol['routes']})
                res_entry.update(metrics)
            except Exception as e:
                res_entry['error'] = str(e)
            results.append(res_entry)

        return jsonify({
            'instance_name': data.get('instance_name'), 'instance_path': data['instance_path'],
            'total_customers': len(c), 'capacity': cap, 'timestamp': datetime.now().isoformat(),
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/export_comparison', methods=['POST'])
def export_comparison():
    data = request.json
    results = data.get('comparison_data', {}).get('results', [])
    fname = f'vrptw_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    if data.get('format') == 'json':
        return Response(json.dumps(data.get('comparison_data'), indent=2), 
                        mimetype='application/json', headers={'Content-Disposition': f'attachment; filename={fname}.json'})
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Solver', 'Distance', 'Time(s)', 'Vehicles', 'Customers', 'Coverage(%)', 'Avg Dist'])
    for r in results:
        if r.get('success'):
            writer.writerow([r.get('solver_name'), r.get('total_distance'), r.get('execution_time'),
                             r.get('vehicles_used'), r.get('customers_served'), r.get('coverage'), r.get('avg_distance')])
    return Response(output.getvalue(), mimetype='text/csv', headers={'Content-Disposition': f'attachment; filename={fname}.csv'})

@app.route('/api/map', methods=['POST'])
def get_map():
    try:
        d, c, _, _ = load_instance(request.json['instance_path'])
        return jsonify({'map_html': get_map_html(create_map(d, c, request.json.get('routes')))})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

    