"""
FastAPI Endpoints for VRPTW Solver Demo.
Dedicated to benchmark solving and simple routing demonstrations.
"""

import os
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from .schemas import (
    Customer, VRPTWInstance,
    SolveRequest, SolutionResult, SolveResponse,
    NodeData, RouteData
)
from .data_loader import parse_solomon_file
from .geo_client import parse_smart_paste, get_route_geometry
from .algorithms.baseline.alns import ALNSSolver
from .algorithms.proposed.hybrid import HybridSolver


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "Solomon"
UI_DIR = BASE_DIR / "src" / "ui"

# App initialization
from contextlib import asynccontextmanager
import httpx

# Global HTTP Client for shared usage across requests
http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    # Startup: Initialize shared HTTP client
    http_client = httpx.AsyncClient(timeout=30.0)
    print("🚀 Application started - HTTP Client initialized")
    try:
        yield
    finally:
        # Shutdown: Close HTTP client cleanly
        if http_client:
            await http_client.aclose()
        print("🛑 Application shut down - HTTP Client closed")

app = FastAPI(title="VRPTW Advanced Planner API", lifespan=lifespan,
    description="Professional VRPTW Solver Research Demo",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for UI
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

# ===== HELPER FUNCTIONS =====

def get_rc_instances() -> List[str]:
    """Retrieve list of Solomon RC instances."""
    if not DATA_DIR.exists():
        return []
    return sorted([f.stem for f in DATA_DIR.glob("rc*.txt")])

def load_instance(name: str) -> VRPTWInstance:
    """Load and parse Solomon benchmark instance."""
    filepath = DATA_DIR / f"{name}.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"Instance not found: {name}")
    return parse_solomon_file(str(filepath))

# ===== API ENDPOINTS =====

@app.get("/api/instances")
def list_instances() -> List[str]:
    """List available Solomon benchmark instances."""
    return get_rc_instances()

@app.post("/api/load")
async def load_instance_data(request: dict):
    """Load instance data for visualization."""
    try:
        instance_name = request.get("instance")
        if not instance_name:
            raise HTTPException(status_code=400, detail="Instance name required")
            
        instance = await run_in_threadpool(load_instance, instance_name)
        
        depot_node = NodeData(
            id=0,
            lat=instance.depot.x,
            lng=instance.depot.y,
            ready_time=instance.depot.ready_time,
            due_time=instance.depot.due_date
        )
        
        customers = [
            NodeData(
                id=c.id,
                lat=c.x,
                lng=c.y,
                demand=c.demand,
                ready_time=c.ready_time,
                due_time=c.due_date,
                service_time=c.service_time
            )
            for c in instance.customers
        ]
        
        return {
            "depot": depot_node,
            "customers": customers,
            "capacity": instance.vehicle_capacity
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/solve")
async def solve_instance(request: SolveRequest) -> SolveResponse:
    """Solve VRPTW using selected algorithms."""
    try:
        if request.instance:
            instance = await run_in_threadpool(load_instance, request.instance)
        elif request.customers and request.depot:
            # Custom Problem
            depot = Customer(
                id=0, x=request.depot.lat, y=request.depot.lng,
                demand=0, ready_time=0, due_date=1440, service_time=0
            )
            customers = [
                Customer(
                    id=c.id, x=c.lat, y=c.lng, demand=c.demand or 10,
                    ready_time=c.ready_time or 0, due_date=c.due_time or 1440,
                    service_time=c.service_time or 15
                )
                for c in request.customers
            ]
            instance = VRPTWInstance(
                name="Custom", depot=depot, customers=customers,
                vehicle_capacity=request.max_vehicles or 100,
                num_vehicles=10
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid request")
            
        if request.max_vehicles:
            instance.num_vehicles = request.max_vehicles
                
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    results = []
    
    for algo in request.algorithms:
        start_time = time.time()
        
        try:
            if algo == "ALNS":
                solver = ALNSSolver(instance, time_limit=request.time_limit)
                solution = await run_in_threadpool(solver.solve)
            elif algo in ("Hybrid", "Proposed"):
                solver = HybridSolver(instance, time_limit=request.time_limit)
                solution = await run_in_threadpool(solver.solve)
            else:
                continue
            
            duration = time.time() - start_time
            
            # Format results for UI
            depot_node = NodeData(id=0, lat=instance.depot.x, lng=instance.depot.y)
            
            routes = []
            for route_idx, route in enumerate(solution.routes):
                if not route:
                    continue
                
                route_nodes = []
                curr_time = 0.0
                prev_id = 0
                all_nodes = [instance.depot] + instance.customers
                
                for nid in route:
                    node = all_nodes[nid]
                    travel_time = instance.distance(prev_id, nid)
                    arrival = curr_time + travel_time
                    start_service = max(arrival, node.ready_time)
                    wait_time = start_service - arrival
                    end_service = start_service + node.service_time
                    
                    route_nodes.append(NodeData(
                        id=nid,
                        lat=node.x,
                        lng=node.y,
                        demand=node.demand,
                        arrival_time=round(arrival, 1),
                        start_service=round(start_service, 1),
                        end_service=round(end_service, 1),
                        wait_time=round(wait_time, 1)
                    ))
                    curr_time = end_service
                    prev_id = nid
                
                # Fetch road geometry for custom problems
                geometry = None
                if not request.instance:
                    coords = [(instance.depot.x, instance.depot.y)] + \
                             [(n.lat, n.lng) for n in route_nodes] + \
                             [(instance.depot.x, instance.depot.y)]
                    from .geo_client import get_route_geometry
                    geometry = await get_route_geometry(coords, client=http_client)

                routes.append(RouteData(
                    nodes=route_nodes,
                    vehicle_id=route_idx + 1,
                    distance=round(sum(instance.distance(route[i], route[i+1]) for i in range(len(route)-1)) + 
                                   instance.distance(0, route[0]) + instance.distance(route[-1], 0), 2),
                    geometry=geometry
                ))
            
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
                time=round(time.time() - start_time, 3), error=str(e)
            ))
    
    return SolveResponse(solutions=results)

@app.post("/api/parse_paste")
async def parse_paste(request: dict):
    """Parse text from Smart Paste Area."""
    text = request.get("text", "")
    from .geo_client import parse_smart_paste, geocode_address
    
    # Simple parsing first
    raw_results = parse_smart_paste(text)
    
    # Perform geocoding concurrently with the shared client
    customers = []
    for entry in raw_results:
        geo = await geocode_address(entry["address"], client=http_client)
        if geo:
            entry["lat"] = geo.lat
            entry["lng"] = geo.lng
        customers.append(entry)
        
    return {"customers": customers, "count": len(customers)}

@app.get("/")
def serve_frontend():
    """Serve the main UI page."""
    return FileResponse(str(UI_DIR / "index.html"))
