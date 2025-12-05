# VRPTW Demo - Ứng dụng Giải bài toán Định tuyến Xe với Ràng buộc Thời gian

Ứng dụng web giải bài toán Vehicle Routing Problem with Time Windows (VRPTW) sử dụng thuật toán tối ưu hóa và trực quan hóa tương tác.

## Tổng quan Dự án

Hệ thống cung cấp giao diện học thuật để giải quyết và trực quan hóa các bài toán VRPTW sử dụng thư viện tối ưu hóa OR-Tools của Google, kết hợp với mô hình DQN.

**Tính năng chính:**
- Giao diện web tương tác với bản đồ Việt Nam
- Hỗ trợ dataset Solomon và Gehring-Homberger
- Tích hợp mô hình DQN Learn-to-Solve
- Trực quan hóa tuyến đường thời gian thực

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

- **Flask** 3.0.0: Web framework
- **OR-Tools** 9.7.0: Thuật toán tối ưu hóa
- **PyTorch** 2.0.0: Mô hình DQN
- **Folium** 0.14.0: Trực quan hóa bản đồ
- **Pandas** 2.0.0: Xử lý dữ liệu
- **Flask-CORS** 4.0.0: Cross-origin support
- **Safetensors** 0.4.0: Lưu trữ mô hình DQN
- **NumPy** 1.24.0: Tính toán số học

## Hướng dẫn Sử dụng

1. **Chọn Instance**: Lựa chọn từ danh sách Solomon (c101, r101, rc101...)
2. **Load Instance**: Nhấn "Load Instance" để hiển thị khách hàng trên bản đồ
3. **Cấu hình Tham số**: Điều chỉnh số lượng xe (mặc định: tự động)
4. **Chạy Giải thuật**: Nhấn "Run Inference" để tối ưu hóa tuyến đường
5. **Xem Kết quả**: Quan sát tuyến đường và metrics trên bản đồ

## Ghi chú

- Instance nhỏ (100 khách hàng): Giải trong vài giây
- Instance lớn (1000+ khách hàng): 1-2 phút
- Hệ thống tự động chọn mô hình DQN hoặc OR-Tools
- Tọa độ được chuyển đổi theo khu vực TP.HCM

---

Dự án nghiên cứu học thuật về bài toán tối ưu hóa logistics với công nghệ web hiện đại.