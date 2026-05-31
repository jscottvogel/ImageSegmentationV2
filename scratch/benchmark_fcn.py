import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Disable cuDNN/MIOpen benchmarking and enabled status to force native PyTorch/HIP implementation
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

from fcn_version import ResNet50FCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

print("Creating model...")
model = ResNet50FCN(num_classes=10).to(device)
model.train()
print("Model created.")

# Dummy input
print("Creating dummy input...")
x = torch.randn(4, 3, 480, 640, device=device)
y = torch.randint(0, 10, (4, 480, 640), device=device)
print("Dummy input created.")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("Starting Warmup...")
for step in range(2):
    print(f"Warmup Step {step} | Forward pass...")
    t0 = time.time()
    out = model(x)
    torch.cuda.synchronize()
    print(f"Warmup Step {step} | Forward pass completed in {time.time() - t0:.4f}s")

    print(f"Warmup Step {step} | Loss computation...")
    loss = F.cross_entropy(out['main_output'], y) + 0.4 * F.cross_entropy(out['aux_output'], y)
    print(f"Warmup Step {step} | Loss = {loss.item():.4f}")

    print(f"Warmup Step {step} | Backward pass...")
    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    print(f"Warmup Step {step} | Backward pass completed in {time.time() - t0:.4f}s")

    print(f"Warmup Step {step} | Optimizer step...")
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    print(f"Warmup Step {step} | Optimizer step completed.")

print("Warmup finished.")
