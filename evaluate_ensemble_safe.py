import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
import gc

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"

from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig, id2color, rgb_to_mask, class_weights
from unet_version import StandardUNet
from fcn_version import ResNet50FCN
from ensemble_inference import safe_morphological_cleanup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_weights_custom(model, path):
    state = torch.load(path, map_location=device, weights_only=True)
    clean_state = {}
    for k, v in state.items():
        k = k.replace('module.', '').replace('_orig_mod.', '')
        clean_state[k] = v
    model.load_state_dict(clean_state, strict=False)
    model.eval()

# 1. Dataset loading (Validation Sample of 20 images)
tr_img = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
tr_msk = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))

np.random.seed(42)
indices = np.random.choice(len(tr_img), 20, replace=False)

# Load ground truth masks and images
print("Loading dataset samples...")
images_t = []
targets = []
for idx in indices:
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
    
    images_t.append(img_tensor)
    targets.append(target)

# We define a function to extract predictions from a model and then unload it
def extract_predictions(model_class, weights_path):
    print(f"Loading {model_class.__name__} from {weights_path}...")
    model = model_class(num_classes=10).to(device)
    load_weights_custom(model, weights_path)
    
    preds = []
    with torch.no_grad():
        for img_tensor in images_t:
            # Run 1.0x single-scale with horizontal flip (TTA)
            # Standard Forward Pass
            out_std = model(img_tensor)
            if isinstance(out_std, dict):
                out_std = out_std['main_output']
            
            # Horizontal Flip Pass
            img_flipped = torch.flip(img_tensor, dims=[3])
            out_flip = model(img_flipped)
            if isinstance(out_flip, dict):
                out_flip = out_flip['main_output']
            out_unflipped = torch.flip(out_flip, dims=[3])
            
            fused = (out_std + out_unflipped) / 2.0
            probs = F.softmax(fused, dim=1).squeeze(0).cpu().numpy() # 10 x 480 x 640
            preds.append(probs)
            
    # Delete model and purge VRAM
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Purged memory. Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    return preds

# Extract predictions sequentially to prevent OOM
pred_dl_best = extract_predictions(CustomDeepLabV3Plus, "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt")
pred_dl_swa = extract_predictions(CustomDeepLabV3Plus, "model_checkpoint/FloodNet_PyTorch/final_swa_smoothed_weights.pt")

pred_unet_best = extract_predictions(StandardUNet, "model_checkpoint/FloodNet_UNet/best_unet_weights.pt")
pred_unet_swa = extract_predictions(StandardUNet, "model_checkpoint/FloodNet_UNet/final_swa_smoothed_unet.pt")

pred_fcn_best = extract_predictions(ResNet50FCN, "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt")

print("\nAll predictions extracted to CPU memory successfully.")

# Setup ensembling weights (converted to numpy on CPU)
w_dl_cw = np.array([1.0, 0.9301, 0.9380, 0.9007, 0.9252, 0.9556, 0.9224, 0.8694, 0.9242, 0.9590], dtype=np.float32).reshape(10, 1, 1)
w_unet_cw = np.array([1.0, 0.9434, 0.9526, 0.9234, 0.9359, 0.9552, 0.9333, 0.8939, 0.9356, 0.9637], dtype=np.float32).reshape(10, 1, 1)
w_fcn_cw = np.array([1.0, 0.9195, 0.9385, 0.9176, 0.9298, 0.9473, 0.9254, 0.8448, 0.9257, 0.9595], dtype=np.float32).reshape(10, 1, 1)

cw_np = class_weights.numpy()
cw_no_0 = cw_np.copy()
cw_no_0[0] = 0.0
cw_no_0 = np.clip(cw_no_0, a_min=0.0, a_max=None)
cw_no_0 = cw_no_0 / (np.sum(cw_no_0) + 1e-6)

# Evaluation function
def evaluate_ensemble(dl_probs, unet_probs, fcn_probs, w_dl, w_unet, w_fcn, min_area=0):
    intersections = np.zeros(10, dtype=np.float64)
    unions = np.zeros(10, dtype=np.float64)
    
    total_w = w_dl + w_unet + w_fcn
    
    for idx in range(len(indices)):
        p_dl = dl_probs[idx]
        p_unet = unet_probs[idx]
        p_fcn = fcn_probs[idx]
        target = targets[idx].squeeze(0).cpu().numpy()
        
        fused = (p_dl * w_dl + p_unet * w_unet + p_fcn * w_fcn) / total_w
        pred_labels = np.argmax(fused, axis=0).astype(np.uint8)
        
        if min_area > 0:
            pred_labels = safe_morphological_cleanup(pred_labels, min_area=min_area)
            
        valid_mask = (target != 255)
        
        for c in range(10):
            pred_c = (pred_labels == c) & valid_mask
            tgt_c = (target == c) & valid_mask
            
            intersections[c] += np.sum(pred_c & tgt_c)
            unions[c] += np.sum(pred_c) + np.sum(tgt_c)
            
    dice = (2. * intersections) / (unions + 1e-6)
    w_dice_no_0 = np.sum(dice * cw_no_0)
    macro_1_9 = np.mean(dice[1:10])
    return w_dice_no_0, macro_1_9

# 1. Sweep SWA vs best weight combinations
weight_options = [
    ("Best Weights Combo", pred_dl_best, pred_unet_best),
    ("SWA Weights Combo", pred_dl_swa, pred_unet_swa),
    ("DeepLab SWA + UNet Best", pred_dl_swa, pred_unet_best),
    ("DeepLab Best + UNet SWA", pred_dl_best, pred_unet_swa)
]

print("\n=== SWEEP 1: SWA VS BEST WEIGHT COMBINATIONS (Class-wise weights, min_area=50) ===")
for name, dl_p, unet_p in weight_options:
    w_dice, macro = evaluate_ensemble(dl_p, unet_p, pred_fcn_best, w_dl_cw, w_unet_cw, w_fcn_cw, min_area=50)
    print(f"Config: {name:25s} | Weighted Dice: {w_dice:.5f} | Macro Mean Dice: {macro:.5f}")

# 2. Sweep ensembling weight strategies (using Best Weights Combo, min_area=50)
weight_strategies = [
    ("Class-wise Unscaled", w_dl_cw, w_unet_cw, w_fcn_cw),
    ("Class-wise Globally-Scaled (0.42, 0.38, 0.20)", w_dl_cw * 0.42, w_unet_cw * 0.38, w_fcn_cw * 0.20),
    ("Global-only (0.42, 0.38, 0.20)", np.full((10, 1, 1), 0.42, dtype=np.float32), np.full((10, 1, 1), 0.38, dtype=np.float32), np.full((10, 1, 1), 0.20, dtype=np.float32))
]

print("\n=== SWEEP 2: ENSEMBLING WEIGHT STRATEGIES (Best weights combo, min_area=50) ===")
for name, w_dl, w_unet, w_fcn in weight_strategies:
    w_dice, macro = evaluate_ensemble(pred_dl_best, pred_unet_best, pred_fcn_best, w_dl, w_unet, w_fcn, min_area=50)
    print(f"Strategy: {name:45s} | Weighted Dice: {w_dice:.5f} | Macro Mean Dice: {macro:.5f}")

# 3. Sweep morphological cleanup min_area (using Best Weights Combo and Class-wise Unscaled weights)
print("\n=== SWEEP 3: MORPHOLOGICAL CLEANUP THRESHOLD Sweep (Best weights, Class-wise weights) ===")
for min_area in [0, 10, 20, 30, 40, 50, 60]:
    w_dice, macro = evaluate_ensemble(pred_dl_best, pred_unet_best, pred_fcn_best, w_dl_cw, w_unet_cw, w_fcn_cw, min_area=min_area)
    print(f"min_area: {min_area:2d} | Weighted Dice: {w_dice:.5f} | Macro Mean Dice: {macro:.5f}")
