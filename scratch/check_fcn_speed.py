import time
import torch
from fcn_version import ResNet50FCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
model = ResNet50FCN(num_classes=10).to(device)
model.eval()

x = torch.randn(1, 3, 480, 640).to(device)

print("Warmup...")
for _ in range(5):
    _ = model(x)

print("Benchmarking...")
start = time.time()
for i in range(10):
    _ = model(x)
torch.cuda.synchronize()
print(f"10 runs took {time.time() - start:.4f}s (Average: {(time.time() - start)/10:.4f}s per forward pass)")
