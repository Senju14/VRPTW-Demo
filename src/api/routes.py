"""FastAPI routes for VRPTW solver."""

import os
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .schemas import SolveRequest, LoadRequest, SolutionResult, SolveResponse, NodeData, RouteData
from ..core import VRPTWInstance, Customer, ALNSSolver, HybridSolver, parse_solomon_file


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "Solomon"
WEB_DIR = BASE_DIR / "src" / "web"

app = FastAPI(title="VRPTW Solver", description="ALNS vs Hybrid DQN+ALNS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def get_rc_instances() -> List[str]:
    """List available RC benchmark instances."""
    if not DATA_DIR.exists():
        return []
    return sorted([
        f.stem for f in DATA_DIR.glob("rc*.txt")
    ])


def load_instance(name: str) -> dict:
    """Load and parse Solomon instance."""
    filepath = DATA_DIR / f"{name}.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"Instance not found: {name}")
    return parse_solomon_file(str(filepath))


def to_node_data(node: dict) -> NodeData:
    """Convert raw node to NodeData."""
    return NodeData(
        id=node.get('id', 0),
        lat=node.get('x', node.get('lat', 0)),
        lng=node.get('y', node.get('lng', 0)),
        demand=node.get('demand', 0),
        ready_time=node.get('ready_time', 0),
        due_time=node.get('due_date', node.get('due_time', 0)),
        service_time=node.get('service_time', 0)
    )


@app.get("/api/instances")
def list_instances() -> List[str]:
    """Get list of available instances."""
    return get_rc_instances()


@app.post("/api/load")
def load_instance_data(request: LoadRequest):
    """Load instance for visualization."""
    try:
        data = load_instance(request.instance)
        
        depot_node = {
            'id': 0,
            'x': data['depot'].x,
            'y': data['depot'].y,
            'ready_time': data['depot'].ready_time,
            'due_time': data['depot'].due_date,
            'service_time': data['depot'].service_time
        }
        
        customers = [
            {
                'id': c.id,
                'x': c.x,
                'y': c.y,
                'demand': c.demand,
                'ready_time': c.ready_time,
                'due_time': c.due_date,
                'service_time': c.service_time
            }
            for c in data['customers']
        ]
        
        return {
            "depot": to_node_data(depot_node),
            "customers": [to_node_data(c) for c in customers],
            "capacity": data['capacity']
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/solve")
def solve_instance(request: SolveRequest) -> SolveResponse:
    """Run solvers on instance."""
    try:
        data = load_instance(request.instance)
        instance = VRPTWInstance(
            name=data['name'],
            depot=data['depot'],
            customers=data['customers'],
            vehicle_capacity=data['capacity'],
            num_vehicles=request.max_vehicles or 25
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    results = []
    
    for algo in request.algorithms:
        start = time.time()
        
        try:
            if algo == "ALNS":
                solver = ALNSSolver(instance, time_limit=15.0)
                solution = solver.solve()
            elif algo in ("Hybrid", "DQN+ALNS"):
                solver = HybridSolver(instance, time_limit=15.0)
                solution = solver.solve()
            else:
                results.append(SolutionResult(
                    algorithm=algo, vehicles=0, distance=0, time=0,
                    error=f"Unknown algorithm: {algo}"
                ))
                continue
            
            duration = time.time() - start
            
            # Format routes for frontend
            depot_node = NodeData(
                id=0, lat=instance.depot.x, lng=instance.depot.y
            )
            
            routes = []
            for route in solution.routes:
                if not route:
                    continue
                nodes = [
                    NodeData(
                        id=nid,
                        lat=instance.customers[nid - 1].x,
                        lng=instance.customers[nid - 1].y,
                        demand=instance.customers[nid - 1].demand
                    )
                    for nid in route
                ]
                routes.append(RouteData(nodes=nodes))
            
            results.append(SolutionResult(
                algorithm=algo,
                vehicles=solution.num_vehicles,
                distance=round(solution.cost, 2),
                time=round(duration, 3),
                routes=routes,
                depot=depot_node
            ))
            
        except Exception as e:
            results.append(SolutionResult(
                algorithm=algo, vehicles=0, distance=0,
                time=time.time() - start, error=str(e)
            ))
    
    return SolveResponse(solutions=results)


@app.get("/")
def serve_frontend():
    """Serve main HTML page."""
    return FileResponse(str(WEB_DIR / "index.html"))
