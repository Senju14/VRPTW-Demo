"""
Routing API Client - Nominatim & OSRM Integration.
Provides geocoding and routing services using free public APIs.
"""

import re
import httpx
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Public API Endpoints
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org"

# Default bounds for geocoding search
VIETNAM_BOUNDS = {
    "viewbox": "102.14,8.18,109.46,23.39", 
    "countrycodes": "vn"
}

@dataclass
class GeocodedAddress:
    """Geocoding result representation."""
    original: str
    lat: float
    lng: float
    display_name: str
    confidence: float

@dataclass
class RouteInfo:
    """Routing information (distance and duration)."""
    distance_km: float
    duration_min: float

async def geocode_address(address: str, client: Optional[httpx.AsyncClient] = None, timeout: float = 10.0) -> Optional[GeocodedAddress]:
    """
    Convert a string address to coordinates using Nominatim.
    """
    params = {
        "q": address,
        "format": "json",
        "limit": 1,
        "countrycodes": VIETNAM_BOUNDS["countrycodes"],
        "viewbox": VIETNAM_BOUNDS["viewbox"],
        "bounded": 1
    }
    headers = {"User-Agent": "VRPTW-Planner/1.0"}
    
    try:
        if client:
            response = await client.get(NOMINATIM_URL, params=params, headers=headers, timeout=timeout)
        else:
            async with httpx.AsyncClient() as c:
                response = await c.get(NOMINATIM_URL, params=params, headers=headers, timeout=timeout)
                
        if response.status_code == 200:
            data = response.json()
            if data:
                result = data[0]
                return GeocodedAddress(
                    original=address,
                    lat=float(result["lat"]),
                    lng=float(result["lon"]),
                    display_name=result.get("display_name", address),
                    confidence=float(result.get("importance", 0.5))
                )
    except Exception:
        pass
    return None

async def get_route_distance(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    client: Optional[httpx.AsyncClient] = None
) -> Optional[RouteInfo]:
    """
    Retrieve real-world distance and duration from OSRM.
    """
    try:
        # OSRM expects: lng,lat format
        coords = f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
        url = f"{OSRM_URL}/route/v1/driving/{coords}"
        
        if client:
            response = await client.get(url, params={"overview": "false"})
        else:
            async with httpx.AsyncClient() as c:
                response = await c.get(url, params={"overview": "false"})
            
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "Ok":
                route = data["routes"][0]
                distance_km = route["distance"] / 1000
                duration_min = route["duration"] / 60
                return RouteInfo(
                    distance_km=round(distance_km, 2),
                    duration_min=round(duration_min, 1)
                )
    except Exception:
        pass
    return None

async def get_route_geometry(
    coords: List[Tuple[float, float]],
    client: Optional[httpx.AsyncClient] = None
) -> Optional[List[Tuple[float, float]]]:
    """
    Retrieve full road geometry for a multi-point route.
    """
    try:
        # lng,lat;lng,lat;...
        coord_str = ";".join([f"{c[1]},{c[0]}" for c in coords])
        url = f"{OSRM_URL}/route/v1/driving/{coord_str}"
        params = {"overview": "full", "geometries": "geojson"}
        
        if client:
            response = await client.get(url, params=params, timeout=15.0)
        else:
            async with httpx.AsyncClient() as c:
                response = await c.get(url, params=params, timeout=15.0)
            
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "Ok":
                # GeoJSON coordinates are [lng, lat]
                geometry = data["routes"][0]["geometry"]["coordinates"]
                return [(coord[1], coord[0]) for coord in geometry]
    except Exception:
        pass
    return None

def parse_smart_paste(text: str) -> List[Dict]:
    """
    Parse input text from the Smart Paste Area.
    Supported formats: Address | Weight | TimeWindow or simple comma-separated.
    """
    results = []
    lines = text.strip().split("\n")
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Default entry
        entry = {
            "id": i + 1,
            "address": line,
            "demand": 10,
            "ready_time": 0,
            "due_time": 1000,
            "service_time": 10
        }
        
        # Try pipe-separated format: Address | Weight | Time
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            entry["address"] = parts[0]
            
            if len(parts) >= 2:
                try:
                    entry["demand"] = float(parts[1])
                except ValueError:
                    pass
            
            if len(parts) >= 3:
                time_part = parts[2]
                time_match = re.search(r"(\d+)(?::\d+)?\s*[-–]\s*(\d+)(?::\d+)?", time_part)
                if time_match:
                    entry["ready_time"] = int(time_match.group(1)) * 60
                    entry["due_time"] = int(time_match.group(2)) * 60
        
        elif "," in line:
            parts = [p.strip() for p in line.split(",")]
            entry["address"] = parts[0]
            
            if len(parts) >= 2:
                try:
                    entry["demand"] = float(parts[1])
                except ValueError:
                    pass
        
        results.append(entry)
    
    return results
