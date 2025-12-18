from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import time
from pathlib import Path

# Import modules
from .data_loader import list_rc_instances, parse_solomon
from .ortools_solver import solve_ortools
from .dummy_solvers import solve_placeholder

# Định vị thư mục frontend tuyệt đối
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "src" / "frontend"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files với đường dẫn tuyệt đối
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
else:
    print(f"⚠️ Warning: Không tìm thấy thư mục frontend tại {FRONTEND_DIR}")

class ComparisonRequest(BaseModel):
    instance: str
    algorithms: List[str]
    max_vehicles: int

class InstanceRequest(BaseModel):
    instance: str

@app.get("/api/instances")
def get_instances():
    """Trả về danh sách file RC"""
    return list_rc_instances()

@app.post("/api/load_instance")
def load_instance_details(req: InstanceRequest):
    """API mới: Trả về dữ liệu thô để vẽ map"""
    try:
        data = parse_solomon(req.instance)
        return {
            "depot": data['depot'],
            "customers": data['customers'],
            "capacity": data['capacity']
        }
    except Exception as e:
        print(f"Error loading instance: {e}")
        raise HTTPException(status_code=404, detail=f"File not found or parse error: {str(e)}")

@app.post("/api/run_comparison")
def run_comparison(req: ComparisonRequest):
    try:
        data = parse_solomon(req.instance)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    results = []
    
    for algo in req.algorithms:
        t0 = time.time()
        result = None
        
        try:
            if algo == "OR-Tools":
                result = solve_ortools(data, req.max_vehicles)
            else:
                result = solve_placeholder(algo, data, req.max_vehicles)
            
            if result:
                result["time"] = time.time() - t0
                results.append(result)
            else:
                results.append({"algorithm": algo, "error": "No solution found"})
                
        except Exception as e:
            print(f"Error algo {algo}: {e}")
            results.append({"algorithm": algo, "error": str(e)})

    return {"solutions": results}

@app.get("/")
def read_root():
    """Trả về file index.html"""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return {"error": f"index.html not found at {index_path}"}
    return FileResponse(str(index_path))
