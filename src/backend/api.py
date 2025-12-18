from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import time
from pathlib import Path

# Import Data Loaders
from .data_loader import list_rc_instances, parse_solomon

# Import Algorithms
from .ortools_solver import solve_ortools
from .utils import VRPTWInstance
from .alns import ALNS
from .dqn import DQNAgent, DQNALNSAgent

# Định vị thư mục frontend
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "src" / "frontend"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

class ComparisonRequest(BaseModel):
    instance: str
    algorithms: List[str]
    max_vehicles: int

class InstanceRequest(BaseModel):
    instance: str

# --- Helper: Convert Solution Object to Frontend JSON ---
def format_solution_for_frontend(solution, raw_data, algo_name, duration):
    """
    Chuyển đổi object Solution (Backend) thành Dictionary (Frontend JSON)
    Backend dùng Node ID (0, 1, 2...), Frontend cần Lat/Lng
    """
    if not solution or not solution.feasible:
        return {
            "algorithm": algo_name,
            "error": "No feasible solution found",
            "vehicles": 0, "distance": 0, "time": duration
        }

    # Map node ID to coordinates from raw_data
    # raw_data['depot'] là node 0
    # raw_data['customers'] là node 1..n
    
    # Tạo map lookup cho nhanh
    node_map = {0: raw_data['depot']}
    for c in raw_data['customers']:
        node_map[c['id']] = c

    formatted_routes = []
    for route_ids in solution.routes:
        nodes_data = []
        for nid in route_ids:
            if nid in node_map:
                nodes_data.append(node_map[nid])
        formatted_routes.append({"nodes": nodes_data})

    return {
        "algorithm": algo_name,
        "vehicles": len(solution.routes),
        "distance": solution.cost,
        "time": duration,
        "routes": formatted_routes,
        "depot": raw_data['depot']
    }

# --- API Endpoints ---

@app.get("/api/instances")
def get_instances():
    return list_rc_instances()

@app.post("/api/load_instance")
def load_instance_details(req: InstanceRequest):
    try:
        data = parse_solomon(req.instance)
        return {
            "depot": data['depot'],
            "customers": data['customers'],
            "capacity": data['capacity']
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/run_comparison")
def run_comparison(req: ComparisonRequest):
    try:
        # 1. Load Data
        raw_data = parse_solomon(req.instance)
        instance_obj = VRPTWInstance(raw_data) # Convert to Matrix format for Algorithms
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Data Error: {str(e)}")

    results = []
    
    for algo in req.algorithms:
        t0 = time.time()
        output = None
        
        try:
            print(f"Running {algo} on {req.instance}...")
            
            # --- Dispatcher ---
            if algo == "OR-Tools":
                # OR-Tools code cũ trả về dict sẵn, dùng luôn
                output = solve_ortools(raw_data, req.max_vehicles)
                if output: output["time"] = time.time() - t0
                
            elif algo == "ALNS":
                solver = ALNS(instance_obj, max_iterations=200) # Demo: 200 iter
                sol = solver.solve()
                duration = time.time() - t0
                output = format_solution_for_frontend(sol, raw_data, algo, duration)

            elif algo == "DQN":
                # Demo: episode thấp để chạy nhanh
                agent = DQNAgent(instance_obj, max_episodes=50) 
                sol = agent.solve()
                duration = time.time() - t0
                output = format_solution_for_frontend(sol, raw_data, algo, duration)

            elif algo == "DQN+ALNS":
                agent = DQNALNSAgent(instance_obj, max_episodes=20)
                sol = agent.solve()
                duration = time.time() - t0
                output = format_solution_for_frontend(sol, raw_data, algo, duration)

            # --- Result Handling ---
            if output:
                results.append(output)
            else:
                results.append({
                    "algorithm": algo, 
                    "error": "No solution returned",
                    "time": time.time() - t0
                })
                
        except Exception as e:
            print(f"Error in {algo}: {e}")
            results.append({"algorithm": algo, "error": str(e)})

    return {"solutions": results}

@app.get("/")
def read_root():
    index_path = FRONTEND_DIR / "index.html"
    return FileResponse(str(index_path))
