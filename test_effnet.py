import torch
from torchvision.models import efficientnet_v2_s

model = efficientnet_v2_s().features
x = torch.randn(1, 3, 512, 512)

for i, m in enumerate(model):
    x = m(x)
    print(i, x.shape)
