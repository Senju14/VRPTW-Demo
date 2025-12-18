import os
import glob
from pathlib import Path

# Tự động định vị thư mục gốc dự án
# src/backend/data_loader.py -> parent=backend -> parent=src -> parent=VRPTW-Demo
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "Solomon"

def list_rc_instances():
    """Liệt kê các file RC trong thư mục Solomon"""
    if not DATA_DIR.exists():
        print(f"⚠️ Warning: Không tìm thấy thư mục data tại: {DATA_DIR}")
        return []
    
    # Lấy tất cả file .txt
    files = list(DATA_DIR.glob("*.txt"))
    # Filter file bắt đầu bằng rc (không phân biệt hoa thường)
    rc_files = [f.name for f in files if f.name.lower().startswith("rc")]
    return sorted(rc_files)

def parse_solomon(filename):
    """Đọc file Solomon"""
    file_path = DATA_DIR / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Instance file not found: {filename}")

    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data = []
    vehicle_capacity = 200 # Default fallback
    
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        
        # Parse customer data lines (Digit start)
        if parts[0].isdigit() and len(parts) >= 7:
            data.append({
                "id": int(parts[0]),
                "lat": float(parts[1]),
                "lng": float(parts[2]),
                "demand": int(parts[3]),
                "ready_time": int(parts[4]),
                "due_time": int(parts[5]),
                "service_time": int(parts[6])
            })
        elif "CAPACITY" in line:
            # Try to get capacity from next line
            idx = lines.index(line)
            try:
                if idx + 1 < len(lines):
                    vehicle_capacity = int(lines[idx+1].strip())
            except:
                pass

    if not data:
        raise ValueError(f"Failed to parse data from {filename}")

    return {
        "name": filename,
        "capacity": vehicle_capacity,
        "depot": data[0],
        "customers": data[1:]
    }
