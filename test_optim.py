import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import torch
import torch.nn as nn
device = torch.device('cuda')
model = nn.Linear(10, 10).to(device)
optimizer = torch.optim.AdamW(model.parameters(), foreach=True)
loss = model(torch.randn(10, 10).to(device)).sum()
loss.backward()
print("Backward successful")
try:
    optimizer.step()
    print("Optimizer step successful")
except Exception as e:
    print(f"Error: {e}")
