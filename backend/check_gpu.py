# check_gpu.py
import torch
print("--- Running GPU Check ---")
print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not found by this Python interpreter.")
print("-------------------------")