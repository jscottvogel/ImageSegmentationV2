import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2

from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig, id2color, rgb_to_mask
from unet_version import StandardUNet
from fcn_version import ResNet50FCN

# Setup validation split
image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))

_, val_image_paths, _, val_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def load_weights_custom(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if 'n_averaged' in state:
        del state['n_averaged']
    clean_state = {}
    for k, v in state.items():
        k = k.replace('module.', '').replace('_orig_mod.', '')
        clean_state[k] = v
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, msk_paths):
        self.img_paths = img_paths
        self.msk_paths = msk_paths
        
    def __len__(self):
        return len(self.img_paths)
        
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.cvtColor(cv2.imread(self.msk_paths[idx]), cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = img.shape[:2]
        
        img_resized = cv2.resize(img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
        img_tensor = torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        
        label = rgb_to_mask(msk, id2color, 10)
        target = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, target, orig_h, orig_w

val_dataset = EvalDataset(val_image_paths, val_mask_paths)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

def normalize_imagenet(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (img_tensor - mean) / std

if __name__ == '__main__':
    print("Loading Best Models...")
    deeplab = CustomDeepLabV3Plus(num_classes=10).to(device)
    deeplab = load_weights_custom(deeplab, "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt", device)
    
    unet = StandardUNet(num_classes=10).to(device)
    unet = load_weights_custom(unet, "model_checkpoint/FloodNet_UNet/best_unet_weights.pt", device)
    
    fcn = ResNet50FCN(num_classes=10).to(device)
    fcn = load_weights_custom(fcn, "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt", device)
    
    # Class weights for hybrid ensemble
    w_dl = torch.tensor([1.0, 0.9301, 0.9380, 0.9007, 0.9252, 0.9556, 0.9224, 0.8694, 0.9242, 0.9590], dtype=torch.float32, device=device).view(10, 1, 1)
    w_unet = torch.tensor([1.0, 0.9434, 0.9526, 0.9234, 0.9359, 0.9552, 0.9333, 0.8939, 0.9356, 0.9637], dtype=torch.float32, device=device).view(10, 1, 1)
    w_fcn = torch.tensor([1.0, 0.9195, 0.9385, 0.9176, 0.9298, 0.9473, 0.9254, 0.8448, 0.9257, 0.9595], dtype=torch.float32, device=device).view(10, 1, 1)
    total_w = w_dl + w_unet + w_fcn
    
    intersection_hybrid = torch.zeros(10, device=device)
    union_hybrid = torch.zeros(10, device=device)
    
    intersection_dl = torch.zeros(10, device=device)
    union_dl = torch.zeros(10, device=device)
    
    intersection_unet = torch.zeros(10, device=device)
    union_unet = torch.zeros(10, device=device)
    
    intersection_fcn = torch.zeros(10, device=device)
    union_fcn = torch.zeros(10, device=device)

    with torch.no_grad():
        for imgs, targets, orig_hs, orig_ws in tqdm(val_loader, desc="Ensemble Evaluation"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            norm_imgs = normalize_imagenet(imgs)
            
            # Predict DeepLab (main output)
            out_dl = deeplab(norm_imgs)
            p_dl = F.softmax(out_dl['main_output'], dim=1)
            
            # Predict UNet
            out_unet = unet(norm_imgs)
            if isinstance(out_unet, dict):
                out_unet = out_unet['main_output']
            p_unet = F.softmax(out_unet, dim=1)
            
            # Predict FCN
            out_fcn = fcn(norm_imgs)
            if isinstance(out_fcn, dict):
                out_fcn = out_fcn['main_output']
            p_fcn = F.softmax(out_fcn, dim=1)
            
            # 1. Evaluate individual models
            # DeepLab
            pred_dl = torch.argmax(p_dl, dim=1)
            valid_mask = (targets != 255)
            targets_safe = torch.where(targets == 255, torch.zeros_like(targets), targets)
            for c in range(10):
                intersection_dl[c] += torch.sum((pred_dl == c) & valid_mask & (targets_safe == c))
                union_dl[c] += torch.sum(((pred_dl == c) | (targets_safe == c)) & valid_mask)
                
            # UNet
            pred_unet = torch.argmax(p_unet, dim=1)
            for c in range(10):
                intersection_unet[c] += torch.sum((pred_unet == c) & valid_mask & (targets_safe == c))
                union_unet[c] += torch.sum(((pred_unet == c) | (targets_safe == c)) & valid_mask)
                
            # FCN
            pred_fcn = torch.argmax(p_fcn, dim=1)
            for c in range(10):
                intersection_fcn[c] += torch.sum((pred_fcn == c) & valid_mask & (targets_safe == c))
                union_fcn[c] += torch.sum(((pred_fcn == c) | (targets_safe == c)) & valid_mask)
            
            # 2. Evaluate Hybrid Ensemble
            # Wait, w_dl is (10, 1, 1), p_dl is (B, 10, H, W).
            # We can multiply them by unsqueezing weights to (1, 10, 1, 1) or reshaping.
            w_dl_b = w_dl.unsqueeze(0) # (1, 10, 1, 1)
            w_unet_b = w_unet.unsqueeze(0)
            w_fcn_b = w_fcn.unsqueeze(0)
            total_w_b = total_w.unsqueeze(0)
            
            fused_probs = (p_dl * w_dl_b + p_unet * w_unet_b + p_fcn * w_fcn_b) / total_w_b
            pred_hybrid = torch.argmax(fused_probs, dim=1)
            
            for c in range(10):
                intersection_hybrid[c] += torch.sum((pred_hybrid == c) & valid_mask & (targets_safe == c))
                union_hybrid[c] += torch.sum(((pred_hybrid == c) | (targets_safe == c)) & valid_mask)
                
    dice_dl = (2. * intersection_dl) / (union_dl + intersection_dl + 1e-6)
    dice_unet = (2. * intersection_unet) / (union_unet + intersection_unet + 1e-6)
    dice_fcn = (2. * intersection_fcn) / (union_fcn + intersection_fcn + 1e-6)
    dice_hybrid = (2. * intersection_hybrid) / (union_hybrid + intersection_hybrid + 1e-6)
    
    print("\n--- Validation Split Average Dice Score (Class 1-9) ---")
    print(f"DeepLab Best: {np.mean(dice_dl[1:].cpu().numpy()):.4f}")
    print(f"UNet Best:    {np.mean(dice_unet[1:].cpu().numpy()):.4f}")
    print(f"FCN Best:     {np.mean(dice_fcn[1:].cpu().numpy()):.4f}")
    print(f"Hybrid:       {np.mean(dice_hybrid[1:].cpu().numpy()):.4f}")
    
    # Let's also print class-wise Dice for Hybrid
    print("\n--- Class-wise Dice for Hybrid Ensemble ---")
    for c in range(10):
        print(f"Class {c}: {dice_hybrid[c].item():.4f}")
