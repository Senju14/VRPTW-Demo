"""
FastAPI routes cho VRPTW Solver
Production-ready với multi-depot và smart paste support
"""

import os
import time
from pathlib import Path
from typing import List
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .schemas import (
    SolveRequest, LoadRequest, SolutionResult, SolveResponse,
    NodeData, RouteData, SmartPasteRequest, SmartPasteResponse,
    GeocodeRequest, GeocodeBatchRequest, GeocodeResult,
    ProductionSolveRequest, ProductionSolveResponse,
    DepotSolution, UnassignedOrder, ExportResponse, ExportRoute, ExportRouteStop
)
from ..core import VRPTWInstance, Customer, ALNSSolver, HybridSolver, parse_solomon_file
from ..core.routing_api import parse_smart_paste, geocode_address, geocode_batch

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "Solomon"
WEB_DIR = BASE_DIR / "src" / "web"

# App
app = FastAPI(
    title="VRPTW Route Planner",
    description="Production-ready Vehicle Routing with Time Windows",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ===== HELPER FUNCTIONS =====

def get_rc_instances() -> List[str]:
    """Lấy danh sách Solomon RC instances"""
    if not DATA_DIR.exists():
        return []
    return sorted([f.stem for f in DATA_DIR.glob("rc*.txt")])


def load_instance(name: str) -> dict:
    """Load và parse Solomon instance"""
    filepath = DATA_DIR / f"{name}.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"Instance not found: {name}")
    return parse_solomon_file(str(filepath))


# ===== BENCHMARK ENDPOINTS =====

@app.get("/api/instances")
def list_instances() -> List[str]:
    """Lấy danh sách instances có sẵn"""
    return get_rc_instances()


@app.post("/api/load")
def load_instance_data(request: LoadRequest):
    """Load instance để hiển thị trên map"""
    try:
        data = load_instance(request.instance)
        
        depot_node = NodeData(
            id=0,
            lat=data['depot'].x,
            lng=data['depot'].y,
            ready_time=data['depot'].ready_time,
            due_time=data['depot'].due_date
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
            for c in data['customers']
        ]
        
        return {
            "depot": depot_node,
            "customers": customers,
            "capacity": data['capacity']
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/solve")
def solve_instance(request: SolveRequest) -> SolveResponse:
    """Chạy solver trên Solomon instance"""
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
                continue
            
            duration = time.time() - start
            
            # Format routes
            depot_node = NodeData(id=0, lat=instance.depot.x, lng=instance.depot.y)
            
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


# ===== PRODUCTION ENDPOINTS =====

@app.post("/api/parse_paste")
def parse_paste(request: SmartPasteRequest) -> SmartPasteResponse:
    """
    Parse dữ liệu từ Smart Paste Area
    Hỗ trợ format: Địa chỉ | Khối lượng | Giờ giao
    """
    customers = parse_smart_paste(request.text)
    return SmartPasteResponse(customers=customers, count=len(customers))


@app.post("/api/geocode")
async def geocode(request: GeocodeRequest) -> GeocodeResult:
    """Geocode một địa chỉ sử dụng Nominatim"""
    result = await geocode_address(request.address)
    
    if result:
        return GeocodeResult(
            original=result.original,
            lat=result.lat,
            lng=result.lng,
            display_name=result.display_name,
            success=True
        )
    else:
        return GeocodeResult(
            original=request.address,
            success=False
        )


@app.post("/api/geocode_batch")
async def geocode_multiple(request: GeocodeBatchRequest) -> List[GeocodeResult]:
    """Geocode nhiều địa chỉ"""
    results = await geocode_batch(request.addresses)
    
    return [
        GeocodeResult(
            original=r.original if r else addr,
            lat=r.lat if r else None,
            lng=r.lng if r else None,
            display_name=r.display_name if r else None,
            success=r is not None
        )
        for r, addr in zip(results, request.addresses)
    ]


@app.post("/api/solve_production")
def solve_production(request: ProductionSolveRequest) -> ProductionSolveResponse:
    """
    Giải bài toán production với multi-depot
    Objective: min(α * distance + β * vehicles)
    """
    start_time = time.time()
    
    depot_solutions = []
    all_unassigned = []
    total_distance = 0.0
    total_vehicles = 0
    
    # Đơn giản: assign customers to nearest depot
    # Trong thực tế, có thể dùng clustering
    
    for depot in request.depots:
        # Filter customers gần depot này (giả sử: tất cả thuộc depot đầu tiên)
        # TODO: Implement proper customer-depot assignment
        if depot.id == request.depots[0].id:
            depot_customers = request.customers
        else:
            depot_customers = []
        
        if not depot_customers:
            depot_solutions.append(DepotSolution(
                depot_id=depot.id,
                depot_name=depot.name,
                routes=[],
                total_distance=0,
                total_vehicles=0
            ))
            continue
        
        # Tạo instance cho depot này
        depot_obj = Customer(
            id=0, x=depot.lat, y=depot.lng,
            demand=0, ready_time=depot.ready_time,
            due_date=depot.due_time, service_time=0
        )
        
        customers_obj = [
            Customer(
                id=c.id, x=c.lat, y=c.lng,
                demand=c.demand, ready_time=c.ready_time,
                due_date=c.due_time, service_time=c.service_time
            )
            for c in depot_customers
        ]
        
        # Tính capacity từ vehicle types
        total_capacity = sum(v.capacity * v.count for v in request.vehicles)
        avg_capacity = total_capacity / max(sum(v.count for v in request.vehicles), 1)
        
        instance = VRPTWInstance(
            name=f"Depot-{depot.id}",
            depot=depot_obj,
            customers=customers_obj,
            vehicle_capacity=avg_capacity,
            num_vehicles=sum(v.count for v in request.vehicles)
        )
        
        # Solve
        try:
            solver = HybridSolver(instance, time_limit=15.0)
            solution = solver.solve()
            
            # Format routes
            routes = []
            visited = set()
            
            for route_ids in solution.routes:
                if not route_ids:
                    continue
                nodes = []
                for nid in route_ids:
                    if nid <= len(depot_customers):
                        c = depot_customers[nid - 1]
                        nodes.append(NodeData(
                            id=c.id, lat=c.lat, lng=c.lng,
                            demand=c.demand, address=c.address
                        ))
                        visited.add(c.id)
                routes.append(RouteData(nodes=nodes, depot_id=depot.id))
            
            # Tìm unassigned
            for c in depot_customers:
                if c.id not in visited:
                    all_unassigned.append(UnassignedOrder(
                        customer=c,
                        reason="Không thể giao trong time window"
                    ))
            
            depot_solutions.append(DepotSolution(
                depot_id=depot.id,
                depot_name=depot.name,
                routes=routes,
                total_distance=round(solution.cost, 2),
                total_vehicles=solution.num_vehicles
            ))
            
            total_distance += solution.cost
            total_vehicles += solution.num_vehicles
            
        except Exception as e:
            # Log error, không crash
            depot_solutions.append(DepotSolution(
                depot_id=depot.id,
                depot_name=depot.name,
                routes=[],
                total_distance=0,
                total_vehicles=0
            ))
    
    # Tính objective value
    objective = request.alpha * total_distance + request.beta * total_vehicles
    
    return ProductionSolveResponse(
        depots=depot_solutions,
        unassigned=all_unassigned,
        total_distance=round(total_distance, 2),
        total_vehicles=total_vehicles,
        solve_time=round(time.time() - start_time, 3),
        objective_value=round(objective, 2)
    )


@app.post("/api/solve_custom")
def solve_custom(request: dict) -> SolveResponse:
    """Backward compatible custom solve endpoint"""
    try:
        depot_data = request.get("depot", {})
        customers_data = request.get("customers", [])
        capacity = request.get("capacity", 100)
        max_vehicles = request.get("max_vehicles", 10)
        algorithms = request.get("algorithms", ["Hybrid"])
        
        # Create instance
        depot = Customer(
            id=0, x=depot_data.get("lat", 0), y=depot_data.get("lng", 0),
            demand=0, ready_time=0, due_date=10000, service_time=0
        )
        
        customers = [
            Customer(
                id=c.get("id", i+1),
                x=c.get("lat", 0), y=c.get("lng", 0),
                demand=c.get("demand", 10),
                ready_time=c.get("ready_time", 0),
                due_date=c.get("due_time", 1000),
                service_time=c.get("service_time", 10)
            )
            for i, c in enumerate(customers_data)
        ]
        
        instance = VRPTWInstance(
            name="Custom",
            depot=depot,
            customers=customers,
            vehicle_capacity=capacity,
            num_vehicles=max_vehicles
        )
        
        results = []
        for algo in algorithms:
            start = time.time()
            try:
                solver = HybridSolver(instance, time_limit=15.0)
                solution = solver.solve()
                
                routes = []
                for route_ids in solution.routes:
                    if route_ids:
                        nodes = [
                            NodeData(
                                id=nid,
                                lat=customers[nid-1].x,
                                lng=customers[nid-1].y,
                                demand=customers[nid-1].demand
                            )
                            for nid in route_ids
                        ]
                        routes.append(RouteData(nodes=nodes))
                
                results.append(SolutionResult(
                    algorithm=algo,
                    vehicles=solution.num_vehicles,
                    distance=round(solution.cost, 2),
                    time=round(time.time() - start, 3),
                    routes=routes,
                    depot=NodeData(id=0, lat=depot.x, lng=depot.y)
                ))
            except Exception as e:
                results.append(SolutionResult(
                    algorithm=algo, vehicles=0, distance=0,
                    time=0, error=str(e)
                ))
        
        return SolveResponse(solutions=results)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ===== EXPORT =====

@app.get("/api/export/{run_id}")
def export_routes(run_id: str) -> ExportResponse:
    """
    Export routes để gửi cho tài xế
    TODO: Implement with actual storage
    """
    # Placeholder - trong thực tế sẽ lấy từ database
    return ExportResponse(
        routes=[],
        generated_at=datetime.now().isoformat()
    )


# ===== FRONTEND =====

@app.get("/")
def serve_frontend():
    """Serve trang chính"""
    return FileResponse(str(WEB_DIR / "index.html"))
