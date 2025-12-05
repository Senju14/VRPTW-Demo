import torch
import sys

def check_gpu_status():
    print("="*40)
    print("KIỂM TRA TRẠNG THÁI GPU (CUDA)")
    print("="*40)
    
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("\n✅ GPU ĐÃ SẴN SÀNG!")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
    else:
        print("\n⚠️  KHÔNG TÌM THẤY GPU")
        print("Hệ thống sẽ chạy bằng CPU.")
        print("Nếu bạn có GPU NVIDIA, hãy cài đặt PyTorch bản CUDA.")

if __name__ == "__main__":
    check_gpu_status()
