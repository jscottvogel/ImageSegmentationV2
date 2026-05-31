import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_FIND_MODE"] = "FAST"
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2
from scipy.optimize import minimize

from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig, id2color, rgb_to_mask
from unet_version import StandardUNet
from fcn_version import ResNet50FCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Setup validation split (same as Keras)
image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))

train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

print(f"Found {len(val_image_paths)} validation images.")

def load_weights_custom(model, path):
    if os.path.exists(path):
        state = torch.load(path, map_location=device, weights_only=True)
        if 'n_averaged' in state:
            del state['n_averaged']
        model.load_state_dict({k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items()}, strict=False)
    model.eval()
    return model

print("Loading models...")
deeplab = CustomDeepLabV3Plus(num_classes=10).to(device)
deeplab = load_weights_custom(deeplab, "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt")

unet = StandardUNet(num_classes=10).to(device)
unet = load_weights_custom(unet, "model_checkpoint/FloodNet_UNet/best_unet_weights.pt")

fcn = ResNet50FCN(num_classes=10).to(device)
fcn = load_weights_custom(fcn, "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt")

# Precompute probabilities on GPU
print("Precomputing probabilities...")
p_dl_list = []
p_unet_list = []
p_fcn_list = []
targets_list = []

normalize_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
normalize_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

with torch.no_grad():
    for idx in tqdm(range(len(val_image_paths))):
        img = cv2.cvtColor(cv2.imread(val_image_paths[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.cvtColor(cv2.imread(val_mask_paths[idx]), cv2.COLOR_BGR2RGB)
        
        label = rgb_to_mask(msk, id2color, 10)
        
        # Resize image for model input
        img_resized = cv2.resize(img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
        img_tensor = torch.tensor(img_resized.transpose(2, 0, 1)[None, ...], dtype=torch.float32, device=device) / 255.0
        img_norm = (img_tensor - normalize_mean) / normalize_std
        
        # Predictions
        out_dl = deeplab(img_norm)
        p_dl = F.softmax(out_dl['main_output'], dim=1)
        
        out_unet = unet(img_norm)
        if isinstance(out_unet, dict):
            out_unet = out_unet['main_output']
        p_unet = F.softmax(out_unet, dim=1)
        
        out_fcn = fcn(img_norm)
        if isinstance(out_fcn, dict):
            out_fcn = out_fcn['main_output']
        p_fcn = F.softmax(out_fcn, dim=1)
        
        # Resize to 256x256 for fast optimization
        p_dl_small = F.interpolate(p_dl, size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        p_unet_small = F.interpolate(p_unet, size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        p_fcn_small = F.interpolate(p_fcn, size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        
        label_small = torch.tensor(cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST), dtype=torch.long, device=device)
        
        p_dl_list.append(p_dl_small)
        p_unet_list.append(p_unet_small)
        p_fcn_list.append(p_fcn_small)
        targets_list.append(label_small)

# Convert to GPU tensors
P_dl = torch.stack(p_dl_list, dim=0).to(device)
P_unet = torch.stack(p_unet_list, dim=0).to(device)
P_fcn = torch.stack(p_fcn_list, dim=0).to(device)
Targets = torch.stack(targets_list, dim=0).to(device)

print("Precomputation finished.")

# Optimize weights
def evaluate_weights(weights):
    # weights is 30 elements: w_dl (10), w_unet (10), w_fcn (10)
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    w_dl = weights_t[0:10].view(1, 10, 1, 1)
    w_unet = weights_t[10:20].view(1, 10, 1, 1)
    w_fcn = weights_t[20:30].view(1, 10, 1, 1)
    
    # Clip to be non-negative
    w_dl = torch.clamp(w_dl, 0.0, 10.0)
    w_unet = torch.clamp(w_unet, 0.0, 10.0)
    w_fcn = torch.clamp(w_fcn, 0.0, 10.0)
    
    # Blend probabilities
    fused = P_dl * w_dl + P_unet * w_unet + P_fcn * w_fcn
    preds = torch.argmax(fused, dim=1)
    
    # Calculate Dice score for class 1-9
    intersection = torch.zeros(10, device=device)
    union = torch.zeros(10, device=device)
    
    valid_mask = (Targets != 255)
    for c in range(10):
        pred_c = (preds == c) & valid_mask
        tgt_c = (Targets == c) & valid_mask
        
        intersection[c] = torch.sum(pred_c & tgt_c)
        union[c] = torch.sum(pred_c | tgt_c)
        
    dice = (2. * intersection) / (union + intersection + 1e-6)
    mean_dice = torch.mean(dice[1:])
    return -mean_dice.item() # minimize negative dice

# Baselines for comparison
print("\nDownsampled Baselines (256x256):")
d_dl = evaluate_weights(np.concatenate([np.ones(10), np.zeros(10), np.zeros(10)]))
print(f"DeepLab: {-d_dl:.4f}")
d_unet = evaluate_weights(np.concatenate([np.zeros(10), np.ones(10), np.zeros(10)]))
print(f"UNet: {-d_unet:.4f}")
d_fcn = evaluate_weights(np.concatenate([np.zeros(10), np.zeros(10), np.ones(10)]))
print(f"FCN: {-d_fcn:.4f}")

# Try equal voting
d_equal = evaluate_weights(np.concatenate([np.ones(10), np.ones(10), np.ones(10)]))
print(f"Equal Ensemble: {-d_equal:.4f}")

# Optimization
print("\nOptimizing weights...")
initial_weights = np.concatenate([
    np.array([1.0, 0.93, 0.94, 0.90, 0.93, 0.96, 0.92, 0.87, 0.92, 0.96]), # dl
    np.array([1.0, 0.94, 0.95, 0.92, 0.94, 0.96, 0.93, 0.89, 0.94, 0.96]), # unet
    np.array([1.0, 0.92, 0.94, 0.92, 0.93, 0.95, 0.93, 0.84, 0.93, 0.96])  # fcn
])

res = minimize(evaluate_weights, initial_weights, method='Powell', options={'maxiter': 20, 'disp': True})

best_weights = res.x
best_weights[0:10] = np.clip(best_weights[0:10], 0.0, 10.0)
best_weights[10:20] = np.clip(best_weights[10:20], 0.0, 10.0)
best_weights[20:30] = np.clip(best_weights[20:30], 0.0, 10.0)

print("\nOptimization results:")
print(f"Best validation Dice (256x256): {-res.fun:.4f}")

print("\nBest weights:")
print("w_dl = np.array([", ", ".join([f"{x:.4f}" for x in best_weights[0:10]]), "])")
print("w_unet = np.array([", ", ".join([f"{x:.4f}" for x in best_weights[10:20]]), "])")
print("w_fcn = np.array([", ", ".join([f"{x:.4f}" for x in best_weights[20:30]]), "])")
