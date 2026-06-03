import torch
print("CUDA/HIP available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    print("Is HIP:", hasattr(torch.version, 'hip') and torch.version.hip is not None)
    if hasattr(torch.version, 'hip'):
        print("HIP Version:", torch.version.hip)
else:
    print("No CUDA/HIP device available to PyTorch.")
