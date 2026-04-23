import torch
from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig
device = torch.device('cuda')
model = CustomDeepLabV3Plus(num_classes=DatasetConfig.NUM_CLASSES).to(device)
model.eval()
img = torch.randn(1, 3, 512, 512, device=device)
with torch.no_grad():
    out = model(img)
print("CNN SUCCESS!")
