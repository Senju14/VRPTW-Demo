# VRPTW Demo - Ứng dụng Giải bài toán Định tuyến Xe với Ràng buộc Thời gian

Ứng dụng web giải bài toán Vehicle Routing Problem with Time Windows (VRPTW) sử dụng thuật toán tối ưu hóa và trực quan hóa tương tác.

## Tổng quan Dự án

Hệ thống cung cấp giao diện học thuật để giải quyết và trực quan hóa các bài toán VRPTW sử dụng nhiều thuật toán tối ưu hóa khác nhau, kết hợp với mô hình DQN.

**Tính năng chính:**
- Giao diện web tương tác với bản đồ Việt Nam
- Hỗ trợ dataset Solomon và Gehring-Homberger
- Hỗ trợ nhiều bộ giải: OR-Tools, ALNS, DQN-only, DQN + ALNS (hybrid)
- Tích hợp mô hình DQN Learn-to-Solve
- Trực quan hóa tuyến đường và so sánh kết quả nhiều solver theo bảng metrics

## Cài đặt và Chạy

### Yêu cầu hệ thống
- Python 3.10 trở lên
- Git
- uv (Khuyên dùng để cài đặt nhanh hơn)

### Các bước cài đặt

#### Cách 1: Sử dụng uv (Khuyên dùng - Tốc độ cao)

1. **Cài đặt uv (nếu chưa có):**
   ```bash
   pip install uv
   ```

2. **Clone và Setup:**
   ```bash
   git clone https://github.com/Senju14/VRPTW-Demo.git
   cd VRPTW-Demo
   
   # Tạo môi trường ảo bằng uv
   uv venv
   
   # Kích hoạt (Windows PowerShell)
   .venv\Scripts\Activate.ps1
   # Kích hoạt (Linux/Mac)
   source .venv/bin/activate
   ```

3. **Cài đặt dependencies:**
   ```bash
   # Cài đặt các thư viện cơ bản
   uv pip install -r requirements.txt
   ```
   
   **Nếu dùng GPU (NVIDIA):**
   ```bash
   # Cài đè PyTorch bản CUDA
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

#### Cách 2: Sử dụng pip truyền thống

1. **Clone repository từ nhánh master:**
   ```bash
   git clone https://github.com/Senju14/VRPTW-Demo.git
   cd VRPTW-Demo
   git checkout master
   ```

2. **Tạo và kích hoạt virtual environment:**
   ```bash
   # Tạo môi trường ảo
   python -m venv .venv
   
   # Kích hoạt (Windows PowerShell)
   .venv\Scripts\Activate.ps1
   
   # Kích hoạt (Linux/Mac)
   source .venv/bin/activate
   ```

3. **Cài đặt dependencies:**
   
   **Lựa chọn 1: Chạy trên CPU (Mặc định)**
   ```bash
   pip install -r requirements.txt
   ```

   **Lựa chọn 2: Chạy trên GPU (Khuyên dùng nếu có NVIDIA GPU)**
   ```bash
   # Cài đặt các thư viện cơ bản trước
   pip install -r requirements.txt
   
   # Sau đó cài đè PyTorch bản CUDA
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

### Kiểm tra và Chạy ứng dụng

1. **Kiểm tra GPU (Tùy chọn):**
   ```bash
   python src/utils/check_gpu.py
   ```

2. **Chạy ứng dụng:**
   ```bash
   python main.py
   ```

3. **Truy cập giao diện:**
   - Mở trình duyệt web: http://127.0.0.1:5000
   - Lưu ý: Không mở trực tiếp file index.html

## Dependencies

- **Flask** (`flask>=3.0.0`): Web backend
- **Flask-CORS** (`flask-cors>=4.0.0`): CORS cho frontend
- **OR-Tools** (`ortools>=9.7.0`): Bộ giải VRPTW baseline
- **PyTorch** (`torch>=2.0.0`): Mô hình DQN
- **Safetensors** (`safetensors>=0.4.0`): Lưu trữ trọng số DQN
- **Folium** (`folium>=0.14.0`): Trực quan hóa bản đồ
- **Pandas** (`pandas>=2.0.0`): Xử lý dữ liệu
- **NumPy** (`numpy>=1.24.0`): Tính toán số học
- **Packaging** (`packaging>=21.0`): Hỗ trợ so sánh phiên bản

> Tất cả các dependency trên đều đã được liệt kê trong `requirements.txt`.

## Cấu trúc thư mục chính

- **Backend**
  - `main.py`: Điểm vào để chạy ứng dụng Flask.
  - `src/backend/app.py`: Khởi tạo Flask app, các API `/api/*`, điều phối chọn solver.
  - `src/backend/solver.py`: File cha, chứa utilities chung (đánh giá solution, DQN agent, v.v.) và export các hàm:
    - `solve_vrptw` (OR-Tools)
    - `solve_alns_vrptw` (ALNS)
    - `solve_dqn_only_vrptw` (DQN-only)
    - `solve_dqn_alns_vrptw` (DQN + ALNS)
  - `src/backend/algorithms/`: Chỉ chứa code từng thuật toán:
    - `ortools_solver.py`
    - `alns_solver.py`
    - `dqn_only_solver.py`
    - `dqn_alns_solver.py`
  - `src/backend/data_loader.py`: Đọc instance Solomon & Gehring-Homberger.
  - `src/backend/visualization.py`: Sinh HTML map bằng Folium.

- **Frontend**
  - `src/frontend/index.html`: Giao diện chính.
  - `src/frontend/static/app.js`: Logic frontend, gọi API backend, vẽ bản đồ, bảng so sánh solver.
  - `src/frontend/static/parser.js`: Parse file instance VRPTW (Solomon / Gehring-Homberger format).
  - `src/frontend/static/style.css`: Giao diện UI.

- **Dữ liệu & mô hình**
  - `data/Solomon/`: Dataset Solomon.
  - `data/Gehring_Homberger/`: Dataset Gehring-Homberger (200–1000 khách).
  - `models/`: Chứa các file mô hình DQN (`*.safetensor`) tương ứng từng instance.

## Hướng dẫn Sử dụng

1. **Chọn Instance**: Lựa chọn từ danh sách Solomon và Gehring-Homberger trong combobox.
2. **Load Instance**: Nhấn **"Load Instance"** để hiển thị khách hàng & depot trên bản đồ.
3. **Chọn solver**: Tick chọn một hoặc nhiều solver (ALNS, DQN, DQN+ALNS, OR-Tools) để so sánh.
4. **Cấu hình Tham số**: Điều chỉnh số lượng xe (nếu để mặc định, hệ thống dùng cấu hình hợp lý cho instance).
5. **Chạy giải thuật**: Nhấn **"Run Comparison"** để chạy các solver đã chọn.
6. **Xem kết quả**:
   - Quan sát tuyến đường của từng solver trên 4 bản đồ.
   - Xem bảng **Comparison Results**: distance, time, số xe, số khách phục vụ, coverage, average distance.

## Ghi chú

- Instance nhỏ (≈100 khách hàng): Thời gian giải thường vài giây.
- Instance lớn (1000+ khách hàng): Có thể mất 1–2 phút tùy solver và cấu hình máy.
- Nếu có sẵn mô hình DQN tương ứng trong `models/`, bạn có thể dùng chế độ **DQN** hoặc **DQN + ALNS**.
- Nếu không có mô hình, bạn vẫn có thể chạy **ALNS** hoặc **OR-Tools** (baseline).
- Tọa độ được chuẩn hóa và ánh xạ về khu vực TP.HCM để dễ quan sát trên bản đồ.

---

Dự án nghiên cứu học thuật về bài toán tối ưu hóa logistics với công nghệ web hiện đại.