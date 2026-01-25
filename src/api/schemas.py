"""
Pydantic schemas cho API requests/responses
Production-ready với multi-depot và smart paste support
"""

from pydantic import BaseModel
from typing import List, Optional


# ===== BASIC TYPES =====

class NodeData(BaseModel):
    """Thông tin node (depot hoặc customer)"""
    id: int
    lat: float
    lng: float
    demand: float = 0
    ready_time: float = 0
    due_time: float = 0
    service_time: float = 0
    address: Optional[str] = None


class DepotData(BaseModel):
    """Thông tin kho/depot"""
    id: int
    name: str
    lat: float
    lng: float
    ready_time: float = 0      # Giờ mở cửa (phút từ 0h)
    due_time: float = 1440     # Giờ đóng cửa (24h = 1440 phút)


class VehicleType(BaseModel):
    """Loại xe với thông số kỹ thuật"""
    id: int
    name: str
    capacity: float
    count: int                  # Số lượng xe loại này
    cost_per_km: float = 5000   # Chi phí/km (VND)


class RouteData(BaseModel):
    """Route với danh sách nodes"""
    nodes: List[NodeData]
    vehicle_id: Optional[int] = None
    depot_id: Optional[int] = None


# ===== BENCHMARK REQUESTS =====

class LoadRequest(BaseModel):
    """Load Solomon instance"""
    instance: str


class SolveRequest(BaseModel):
    """Solve Solomon benchmark"""
    instance: str
    algorithms: List[str]
    max_vehicles: Optional[int] = None


# ===== PRODUCTION REQUESTS =====

class SmartPasteRequest(BaseModel):
    """Parse dữ liệu từ Smart Paste Area"""
    text: str


class SmartPasteResponse(BaseModel):
    """Kết quả parse"""
    customers: List[dict]
    count: int


class GeocodeRequest(BaseModel):
    """Geocode một địa chỉ"""
    address: str


class GeocodeBatchRequest(BaseModel):
    """Geocode nhiều địa chỉ"""
    addresses: List[str]


class GeocodeResult(BaseModel):
    """Kết quả geocode"""
    original: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    display_name: Optional[str] = None
    success: bool = False


class ProductionSolveRequest(BaseModel):
    """
    Request giải bài toán production
    Hỗ trợ multi-depot, multi-vehicle type
    """
    depots: List[DepotData]
    vehicles: List[VehicleType]
    customers: List[NodeData]
    
    # Optimization weights
    alpha: float = 1.0          # Weight cho distance
    beta: float = 100.0         # Weight cho số xe
    
    # Vietnam context
    current_hour: int = 12      # Giờ hiện tại (để tính congestion)
    use_osrm: bool = False      # Dùng OSRM cho real distance


class UnassignedOrder(BaseModel):
    """Đơn hàng không thể giao"""
    customer: NodeData
    reason: str


class DepotSolution(BaseModel):
    """Giải pháp cho một depot"""
    depot_id: int
    depot_name: str
    routes: List[RouteData]
    total_distance: float
    total_vehicles: int


class ProductionSolveResponse(BaseModel):
    """
    Response giải bài toán production
    """
    depots: List[DepotSolution]
    unassigned: List[UnassignedOrder]
    total_distance: float
    total_vehicles: int
    solve_time: float
    objective_value: float


# ===== SOLUTION RESULT =====

class SolutionResult(BaseModel):
    """Kết quả thuật toán"""
    algorithm: str
    vehicles: int
    distance: float
    time: float
    routes: Optional[List[RouteData]] = None
    depot: Optional[NodeData] = None
    error: Optional[str] = None


class SolveResponse(BaseModel):
    """Response chứa các solutions"""
    solutions: List[SolutionResult]


# ===== EXPORT =====

class ExportRouteStop(BaseModel):
    """Một điểm dừng trong route"""
    order: int
    customer_id: int
    address: str
    lat: float
    lng: float
    demand: float
    eta: str
    time_window: str


class ExportRoute(BaseModel):
    """Route để export cho tài xế"""
    vehicle_id: int
    vehicle_name: str
    depot: str
    total_distance: float
    total_stops: int
    stops: List[ExportRouteStop]


class ExportResponse(BaseModel):
    """Export data"""
    routes: List[ExportRoute]
    generated_at: str
