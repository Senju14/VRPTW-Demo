import uvicorn
import os
import sys

# Thêm src vào path để import module dễ dàng
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

if __name__ == "__main__":
    # Chạy server tại port 8000
    uvicorn.run(
        "src.backend.api:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True
    )
    