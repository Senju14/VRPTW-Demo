"""
Routing API Client - Nominatim & OSRM Integration
Tích hợp API miễn phí cho geocoding và routing
"""

import re
import httpx
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# API Endpoints (miễn phí, public)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OSRM_URL = "https://router.project-osrm.org"

# Vietnam-specific settings
VIETNAM_BOUNDS = {
    "viewbox": "102.14,8.18,109.46,23.39",  # Vietnam bounding box
    "countrycodes": "vn"
}

# Hệ số tắc đường giờ cao điểm (Vietnam)
CONGESTION_COEFFICIENTS = {
    "morning_peak": (7, 9, 1.5),    # 7-9h: x1.5
    "evening_peak": (17, 19, 1.8),  # 17-19h: x1.8
    "normal": 1.0
}


@dataclass
class GeocodedAddress:
    """Kết quả geocoding"""
    original: str
    lat: float
    lng: float
    display_name: str
    confidence: float


@dataclass
class RouteInfo:
    """Thông tin tuyến đường"""
    distance_km: float
    duration_min: float
    duration_with_traffic: float  # Có tính tắc đường


async def geocode_address(address: str, timeout: float = 10.0) -> Optional[GeocodedAddress]:
    """
    Chuyển địa chỉ thành tọa độ sử dụng Nominatim
    
    Args:
        address: Địa chỉ cần geocode (VD: "123 Nguyễn Huệ, Q1, TPHCM")
        timeout: Thời gian chờ tối đa
    
    Returns:
        GeocodedAddress hoặc None nếu không tìm thấy
    """
    try:
        async with httpx.AsyncClient() as client:
            params = {
                "q": address,
                "format": "json",
                "limit": 1,
                "countrycodes": VIETNAM_BOUNDS["countrycodes"],
                "viewbox": VIETNAM_BOUNDS["viewbox"],
                "bounded": 1
            }
            headers = {"User-Agent": "VRPTW-Planner/1.0"}
            
            response = await client.get(
                NOMINATIM_URL, 
                params=params, 
                headers=headers,
                timeout=timeout
            )
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data:
                return None
            
            result = data[0]
            return GeocodedAddress(
                original=address,
                lat=float(result["lat"]),
                lng=float(result["lon"]),
                display_name=result.get("display_name", address),
                confidence=float(result.get("importance", 0.5))
            )
    except Exception:
        return None


async def geocode_batch(addresses: List[str]) -> List[Optional[GeocodedAddress]]:
    """Geocode nhiều địa chỉ cùng lúc"""
    import asyncio
    results = await asyncio.gather(*[geocode_address(addr) for addr in addresses])
    return list(results)


async def get_route_distance(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    hour: int = 12
) -> Optional[RouteInfo]:
    """
    Lấy khoảng cách & thời gian thực tế từ OSRM
    
    Args:
        origin: (lat, lng) điểm xuất phát
        destination: (lat, lng) điểm đến
        hour: Giờ trong ngày (để tính hệ số tắc đường)
    
    Returns:
        RouteInfo với distance và duration
    """
    try:
        # OSRM sử dụng format: lng,lat
        coords = f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
        url = f"{OSRM_URL}/route/v1/driving/{coords}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"overview": "false"})
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            if data.get("code") != "Ok":
                return None
            
            route = data["routes"][0]
            distance_km = route["distance"] / 1000
            duration_min = route["duration"] / 60
            
            # Áp dụng hệ số tắc đường
            congestion = get_congestion_coefficient(hour)
            duration_with_traffic = duration_min * congestion
            
            return RouteInfo(
                distance_km=round(distance_km, 2),
                duration_min=round(duration_min, 1),
                duration_with_traffic=round(duration_with_traffic, 1)
            )
    except Exception:
        return None


def get_congestion_coefficient(hour: int) -> float:
    """
    Lấy hệ số tắc đường theo giờ (Vietnam context)
    
    Args:
        hour: Giờ trong ngày (0-23)
    
    Returns:
        Hệ số nhân thời gian
    """
    morning_start, morning_end, morning_coef = CONGESTION_COEFFICIENTS["morning_peak"]
    evening_start, evening_end, evening_coef = CONGESTION_COEFFICIENTS["evening_peak"]
    
    if morning_start <= hour < morning_end:
        return morning_coef
    elif evening_start <= hour < evening_end:
        return evening_coef
    else:
        return CONGESTION_COEFFICIENTS["normal"]


def parse_smart_paste(text: str) -> List[Dict]:
    """
    Parse dữ liệu từ Smart Paste Area
    
    Hỗ trợ các format:
    1. Địa chỉ | Khối lượng | Giờ bắt đầu - Giờ kết thúc
    2. Địa chỉ, Khối lượng, Giờ giao
    3. Mỗi dòng một địa chỉ (khối lượng mặc định = 10)
    
    Args:
        text: Nội dung paste
    
    Returns:
        List các dict với keys: address, demand, ready_time, due_time
    """
    results = []
    lines = text.strip().split("\n")
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Mặc định
        entry = {
            "id": i + 1,
            "address": line,
            "demand": 10,
            "ready_time": 0,
            "due_time": 1000,
            "service_time": 10
        }
        
        # Thử parse format: Địa chỉ | Khối lượng | Giờ
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
                # Parse "8:00 - 12:00" hoặc "8-12"
                time_match = re.search(r"(\d+)(?::\d+)?\s*[-–]\s*(\d+)(?::\d+)?", time_part)
                if time_match:
                    entry["ready_time"] = int(time_match.group(1)) * 60
                    entry["due_time"] = int(time_match.group(2)) * 60
        
        # Thử parse format CSV: Địa chỉ, Khối lượng, Giờ
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


# Synchronous wrappers for non-async contexts
def geocode_address_sync(address: str) -> Optional[GeocodedAddress]:
    """Sync wrapper cho geocode_address"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    return loop.run_until_complete(geocode_address(address))


def get_route_distance_sync(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    hour: int = 12
) -> Optional[RouteInfo]:
    """Sync wrapper cho get_route_distance"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    return loop.run_until_complete(get_route_distance(origin, destination, hour))
