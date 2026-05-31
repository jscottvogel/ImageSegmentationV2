import torch
import os
from synergistic_model import FloodNetSynergisticNet

model = FloodNetSynergisticNet()
device = 'cpu'

checkpoint_dir = "model_checkpoint"
unet_path = os.path.join(checkpoint_dir, "FloodNet_UNet", "best_unet_weights.pt")
unet_state = torch.load(unet_path, map_location=device, weights_only=True)
unet_state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in unet_state.items()}

new_state = {}
for k, v in unet_state.items():
    if k.startswith("backbone."):
        new_state[k] = v
    elif k.startswith("up"):
        new_state[k.replace("up", "unet_up")] = v
    elif k.startswith("outc."):
        new_state[k.replace("outc.", "unet_outc.")] = v
        
missing, unexpected = model.load_state_dict(new_state, strict=False)
print("=== MISSING KEYS ===")
for m in sorted(list(missing)):
    print(m)
