import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import glob
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig, id2color, rgb_to_mask
from unet_version import StandardUNet
from fcn_version import ResNet50FCN
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading models...")
# 1. DeepLab
model_dl = CustomDeepLabV3Plus(num_classes=10).to(device)
dl_weights = "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt"
if os.path.exists(dl_weights):
    state = torch.load(dl_weights, map_location=device, weights_only=True)
    model_dl.load_state_dict({k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items()}, strict=False)
model_dl.eval()

# 2. UNet
model_unet = StandardUNet(num_classes=10).to(device)
unet_weights = "model_checkpoint/FloodNet_UNet/best_unet_weights.pt"
if os.path.exists(unet_weights):
    state = torch.load(unet_weights, map_location=device, weights_only=True)
    model_unet.load_state_dict({k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items()}, strict=False)
model_unet.eval()

# 3. FCN
model_fcn = ResNet50FCN(num_classes=10).to(device)
fcn_weights = "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt"
if os.path.exists(fcn_weights):
    state = torch.load(fcn_weights, map_location=device, weights_only=True)
    model_fcn.load_state_dict({k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items()}, strict=False)
model_fcn.eval()

print("Models loaded.")

# Dataset
tr_img = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
tr_msk = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))

# Random sample of 150 images
np.random.seed(42)
indices = np.random.choice(len(tr_img), 150, replace=False)

def eval_model(model, model_type='dl'):
    intersection = torch.zeros(10, device=device)
    union = torch.zeros(10, device=device)
    
    with torch.no_grad():
        for idx in tqdm(indices, desc=f"Evaluating {model_type}"):
            img = cv2.cvtColor(cv2.imread(tr_img[idx]), cv2.COLOR_BGR2RGB)
            msk = cv2.cvtColor(cv2.imread(tr_msk[idx]), cv2.COLOR_BGR2RGB)
            
            img = cv2.resize(img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            msk = cv2.resize(msk, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            
            label = rgb_to_mask(msk, id2color, 10)
            
            img_tensor = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            target = torch.tensor(label, dtype=torch.long, device=device).unsqueeze(0)
            
            if model_type == 'dl':
                out = model(img_tensor)['main_output']
            else:
                out = model(img_tensor)
                if isinstance(out, dict):
                    if 'main_output' in out:
                        out = out['main_output']
                    elif 'out' in out:
                        out = out['out']
                
            pred_labels = torch.argmax(out, dim=1)
            
            # mask out 255
            valid_mask = (target != 255)
            target_safe = torch.where(target == 255, torch.zeros_like(target), target)
            
            for c in range(10):
                if c == 0: continue # ignore background usually, but let's just do it
                pred_c = (pred_labels == c) & valid_mask
                tgt_c = (target_safe == c) & valid_mask
                
                intersection[c] += torch.sum(pred_c & tgt_c)
                union[c] += torch.sum(pred_c | tgt_c)
                
    dice = (2. * intersection) / (union + intersection + 1e-6)
    return dice.cpu().numpy()

dice_dl = eval_model(model_dl, 'dl')
dice_unet = eval_model(model_unet, 'unet')
dice_fcn = eval_model(model_fcn, 'fcn')

print("\n--- CLASS-WISE HARD DICE ---")
for c in range(1, 10):
    print(f"Class {c:2d} | DeepLab: {dice_dl[c]:.4f} | UNet: {dice_unet[c]:.4f} | FCN: {dice_fcn[c]:.4f}")
