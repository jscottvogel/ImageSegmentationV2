import torch
from crfseg import CRF

# Create a random image and prob map
img = torch.rand(1, 3, 512, 512)
probs = torch.rand(1, 10, 512, 512)

crf = CRF(n_spatial_dims=2)
# The output is log-probabilities or probabilities?
out = crf(img, probs)
print("CRF Output shape:", out.shape)
