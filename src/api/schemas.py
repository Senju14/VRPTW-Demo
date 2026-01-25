"""Pydantic schemas for API requests/responses."""

from pydantic import BaseModel
from typing import List, Optional


class SolveRequest(BaseModel):
    """Request to solve VRPTW instance."""
    instance: str
    algorithms: List[str]  # ["ALNS", "Hybrid"]
    max_vehicles: Optional[int] = None


class LoadRequest(BaseModel):
    """Request to load instance data."""
    instance: str


class NodeData(BaseModel):
    """Node information for frontend."""
    id: int
    lat: float
    lng: float
    demand: float = 0
    ready_time: float = 0
    due_time: float = 0
    service_time: float = 0


class RouteData(BaseModel):
    """Route with nodes."""
    nodes: List[NodeData]


class SolutionResult(BaseModel):
    """Algorithm solution result."""
    algorithm: str
    vehicles: int
    distance: float
    time: float
    routes: Optional[List[RouteData]] = None
    depot: Optional[NodeData] = None
    error: Optional[str] = None


class SolveResponse(BaseModel):
    """Response containing all solutions."""
    solutions: List[SolutionResult]


class InstanceData(BaseModel):
    """Instance data for visualization."""
    depot: NodeData
    customers: List[NodeData]
    capacity: float
