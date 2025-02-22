import torch
print(torch.__version__)  # Check PyTorch version
print(torch.version.cuda)  # Check CUDA version PyTorch was built with
print(torch.cuda.is_available())  # Ensure CUDA is detected
torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.device_count()
torch.cuda.get_device_name(0)
