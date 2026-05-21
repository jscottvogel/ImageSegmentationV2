import torch
path = "model_checkpoint/FloodNet_Meta/meta_layer_weights.pt"
sd = torch.load(path, map_location='cpu')
# Zero out all weights connecting the FCN (channels 10 to 19) to the output
sd['weight'][:, 10:20, :, :] = 0.0
# Slightly boost UNet and DeepLab to compensate
sd['weight'][:, 0:10, :, :] *= 1.5
sd['weight'][:, 20:30, :, :] *= 1.5
torch.save(sd, path)
print("FCN surgically removed from Meta-Learner weights!")
