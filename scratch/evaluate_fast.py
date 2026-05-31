import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF_f
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
import cv2

from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig, id2color, rgb_to_mask
from unet_version import StandardUNet
from fcn_version import ResNet50FCN

# Setup validation split (same as Keras)
image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))

train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use only 20 images for speed
val_image_paths = val_image_paths[:20]
val_mask_paths = val_mask_paths[:20]

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

# 9 TTA Augmentations
def apply_tta_mode(img_tensor, mode):
    if mode == "hflip":
        return TF_f.hflip(img_tensor)
    elif mode == "vflip":
        return TF_f.vflip(img_tensor)
    elif mode == "brightness":
        return torch.clamp(img_tensor + 0.1, 0.0, 1.0)
    elif mode == "contrast":
        mean = img_tensor.mean(dim=(2, 3), keepdim=True)
        return torch.clamp((img_tensor - mean) * 1.2 + mean, 0.0, 1.0)
    elif mode == "gamma":
        return torch.clamp(img_tensor ** 0.9, 0.0, 1.0)
    elif mode == "saturation":
        squeezed = img_tensor.squeeze(0)
        saturated = TF_f.adjust_saturation(squeezed, 1.2)
        return saturated.unsqueeze(0)
    elif mode == "hue":
        squeezed = img_tensor.squeeze(0)
        hue_adj = TF_f.adjust_hue(squeezed, 0.05)
        return hue_adj.unsqueeze(0)
    elif mode == "gaussian_noise":
        noise = torch.randn_like(img_tensor) * 0.01
        return torch.clamp(img_tensor + noise, 0.0, 1.0)
    else:
        return img_tensor

def invert_tta_mode(probs, mode):
    if mode == "hflip":
        return TF_f.hflip(probs)
    elif mode == "vflip":
        return TF_f.vflip(probs)
    else:
        return probs

def normalize_imagenet(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (img_tensor - mean) / std

def get_deeplab_fused(model, img_tensor, mode_fusion=True, edge_sharpening=True):
    out = model(img_tensor)
    if not mode_fusion:
        return F.softmax(out['main_output'], dim=1), None
        
    main_p = F.softmax(out['main_output'], dim=1)
    side_p = F.softmax(out['side_output'], dim=1)
    mid_p = F.softmax(out['mid_output'], dim=1)
    high_p = F.softmax(out['high_output'], dim=1)
    low_p = F.softmax(out['low_output'], dim=1)
    mid_high_p = F.softmax(out['mid_high_output'], dim=1)
    
    combined = 0.4 * main_p + 0.2 * side_p + 0.2 * mid_p + 0.2 * high_p + 0.2 * low_p + 0.2 * mid_high_p
    
    if edge_sharpening:
        edge_p = torch.sigmoid(out['edge_output'])
        combined *= (1.0 + edge_p)
        
    combined /= (combined.sum(dim=1, keepdim=True) + 1e-6)
    return combined, edge_p if edge_sharpening else None

def crf_like_smoothing(probs, sigma=0.1):
    num_classes, H, W = probs.shape
    smoothed = np.zeros_like(probs)
    for c in range(num_classes):
        smoothed[c] = gaussian_filter(probs[c], sigma=sigma)
    smoothed /= (np.sum(smoothed, axis=0, keepdims=True) + 1e-8)
    return smoothed

def evaluate_configuration(model, model_type='dl', mode_fusion=False, edge_sharpening=False, use_tta=False, use_smoothing=False):
    intersection = torch.zeros(10, device=device)
    union = torch.zeros(10, device=device)
    
    tta_modes = ["none", "hflip", "vflip", "brightness", "contrast", "gamma", "saturation", "hue", "gaussian_noise"] if use_tta else ["none"]
    
    with torch.no_grad():
        for idx in range(len(val_image_paths)):
            img = cv2.cvtColor(cv2.imread(val_image_paths[idx]), cv2.COLOR_BGR2RGB)
            msk = cv2.cvtColor(cv2.imread(val_mask_paths[idx]), cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
            
            label = rgb_to_mask(msk, id2color, 10)
            target = torch.tensor(label, dtype=torch.long, device=device)
            
            img_resized = cv2.resize(img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_01 = torch.tensor(img_resized.transpose(2, 0, 1)[None, ...], dtype=torch.float32, device=device) / 255.0
            
            preds_all = []
            confs_all = []
            
            for mode in tta_modes:
                aug_img = apply_tta_mode(img_01, mode)
                norm_img = normalize_imagenet(aug_img)
                
                if model_type == 'dl':
                    pred_probs, _ = get_deeplab_fused(model, norm_img, mode_fusion, edge_sharpening)
                else:
                    out = model(norm_img)
                    if isinstance(out, dict):
                        out = out['main_output']
                    pred_probs = F.softmax(out, dim=1)
                
                restored_probs = invert_tta_mode(pred_probs, mode)
                
                if use_tta:
                    conf = restored_probs.max(dim=1, keepdim=True)[0]
                    preds_all.append(restored_probs * conf)
                    confs_all.append(conf)
                else:
                    preds_all.append(restored_probs)
                    
            if use_tta:
                combined_probs = sum(preds_all) / (sum(confs_all) + 1e-6)
            else:
                combined_probs = preds_all[0]
                
            combined_probs = F.interpolate(combined_probs, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0)
            probs_np = combined_probs.cpu().numpy()
            
            if use_smoothing:
                probs_np = crf_like_smoothing(probs_np, sigma=0.1)
                
            pred_labels = torch.tensor(np.argmax(probs_np, axis=0), device=device)
            
            valid_mask = (target != 255)
            target_safe = torch.where(target == 255, torch.zeros_like(target), target)
            
            for c in range(10):
                pred_c = (pred_labels == c) & valid_mask
                tgt_c = (target_safe == c) & valid_mask
                intersection[c] += torch.sum(pred_c & tgt_c)
                union[c] += torch.sum(pred_c | tgt_c)
                
    dice = (2. * intersection) / (union + intersection + 1e-6)
    return dice.cpu().numpy()

if __name__ == '__main__':
    print("Loading models...")
    deeplab = CustomDeepLabV3Plus(num_classes=10).to(device)
    deeplab = load_weights_custom(deeplab, "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt", device)
    
    print("\n--- DEEPLAB FAST EXPERIMENTS (20 images) ---")
    
    dice_base = evaluate_configuration(deeplab, 'dl', mode_fusion=False, edge_sharpening=False, use_tta=False, use_smoothing=False)
    print(f"1. DeepLab Main Output Baseline | Class 1-9 Avg Dice: {np.mean(dice_base[1:]):.4f}")
    
    dice_tta = evaluate_configuration(deeplab, 'dl', mode_fusion=False, edge_sharpening=False, use_tta=True, use_smoothing=False)
    print(f"2. DeepLab Main Output + TTA | Class 1-9 Avg Dice: {np.mean(dice_tta[1:]):.4f}")
    
    dice_smooth = evaluate_configuration(deeplab, 'dl', mode_fusion=False, edge_sharpening=False, use_tta=False, use_smoothing=True)
    print(f"3. DeepLab Main Output + Smooth | Class 1-9 Avg Dice: {np.mean(dice_smooth[1:]):.4f}")
    
    dice_tta_smooth = evaluate_configuration(deeplab, 'dl', mode_fusion=False, edge_sharpening=False, use_tta=True, use_smoothing=True)
    print(f"4. DeepLab Main Output + TTA + Smooth | Class 1-9 Avg Dice: {np.mean(dice_tta_smooth[1:]):.4f}")

    print("\nLoading UNet...")
    unet = StandardUNet(num_classes=10).to(device)
    unet = load_weights_custom(unet, "model_checkpoint/FloodNet_UNet/best_unet_weights.pt", device)
    
    print("\n--- UNET FAST EXPERIMENTS (20 images) ---")
    dice_unet_base = evaluate_configuration(unet, 'unet', use_tta=False, use_smoothing=False)
    print(f"1. UNet Baseline | Class 1-9 Avg Dice: {np.mean(dice_unet_base[1:]):.4f}")
    
    dice_unet_tta = evaluate_configuration(unet, 'unet', use_tta=True, use_smoothing=False)
    print(f"2. UNet + TTA | Class 1-9 Avg Dice: {np.mean(dice_unet_tta[1:]):.4f}")
    
    dice_unet_smooth = evaluate_configuration(unet, 'unet', use_tta=False, use_smoothing=True)
    print(f"3. UNet + Smooth | Class 1-9 Avg Dice: {np.mean(dice_unet_smooth[1:]):.4f}")
    
    dice_unet_tta_smooth = evaluate_configuration(unet, 'unet', use_tta=True, use_smoothing=True)
    print(f"4. UNet + TTA + Smooth | Class 1-9 Avg Dice: {np.mean(dice_unet_tta_smooth[1:]):.4f}")

    print("\nLoading FCN...")
    fcn = ResNet50FCN(num_classes=10).to(device)
    fcn = load_weights_custom(fcn, "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt", device)
    
    print("\n--- FCN FAST EXPERIMENTS (20 images) ---")
    dice_fcn_base = evaluate_configuration(fcn, 'fcn', use_tta=False, use_smoothing=False)
    print(f"1. FCN Baseline | Class 1-9 Avg Dice: {np.mean(dice_fcn_base[1:]):.4f}")
    
    dice_fcn_tta = evaluate_configuration(fcn, 'fcn', use_tta=True, use_smoothing=False)
    print(f"2. FCN + TTA | Class 1-9 Avg Dice: {np.mean(dice_fcn_tta[1:]):.4f}")
    
    dice_fcn_smooth = evaluate_configuration(fcn, 'fcn', use_tta=False, use_smoothing=True)
    print(f"3. FCN + Smooth | Class 1-9 Avg Dice: {np.mean(dice_fcn_smooth[1:]):.4f}")
    
    dice_fcn_tta_smooth = evaluate_configuration(fcn, 'fcn', use_tta=True, use_smoothing=True)
    print(f"4. FCN + TTA + Smooth | Class 1-9 Avg Dice: {np.mean(dice_fcn_tta_smooth[1:]):.4f}")

    # Evaluate Hybrid and Meta-Learner ensembles on the 20 images
    print("\nEvaluating Ensembles (no TTA, no DeepLab multi-scale fusion)...")
    
    w_dl = np.array([1.0, 0.9301, 0.9380, 0.9007, 0.9252, 0.9556, 0.9224, 0.8694, 0.9242, 0.9590], dtype=np.float32).reshape(10, 1, 1)
    w_unet = np.array([1.0, 0.9434, 0.9526, 0.9234, 0.9359, 0.9552, 0.9333, 0.8939, 0.9356, 0.9637], dtype=np.float32).reshape(10, 1, 1)
    w_fcn = np.array([1.0, 0.9195, 0.9385, 0.9176, 0.9298, 0.9473, 0.9254, 0.8448, 0.9257, 0.9595], dtype=np.float32).reshape(10, 1, 1)
    total_w = w_dl + w_unet + w_fcn
    
    intersection_hybrid = np.zeros(10)
    union_hybrid = np.zeros(10)
    
    intersection_meta = np.zeros(10)
    union_meta = np.zeros(10)
    
    meta_weights_path = "model_checkpoint/FloodNet_Meta/meta_layer_weights.pt"
    meta_layer = None
    if os.path.exists(meta_weights_path):
        meta_state = torch.load(meta_weights_path, map_location='cpu', weights_only=True)
        meta_layer = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=1)
        meta_layer.load_state_dict(meta_state)
        meta_layer.eval()
        meta_layer = meta_layer.to(device)

    with torch.no_grad():
        for idx in tqdm(range(len(val_image_paths))):
            img = cv2.cvtColor(cv2.imread(val_image_paths[idx]), cv2.COLOR_BGR2RGB)
            msk = cv2.cvtColor(cv2.imread(val_mask_paths[idx]), cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
            
            label = rgb_to_mask(msk, id2color, 10)
            target = torch.tensor(label, dtype=torch.long, device=device)
            valid_mask = (target != 255)
            target_safe = torch.where(target == 255, torch.zeros_like(target), target)
            
            img_resized = cv2.resize(img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_01 = torch.tensor(img_resized.transpose(2, 0, 1)[None, ...], dtype=torch.float32, device=device) / 255.0
            norm_img = normalize_imagenet(img_01)
            
            # Predict DeepLab (main output)
            out_dl = deeplab(norm_img)
            p_dl = F.softmax(out_dl['main_output'], dim=1)
            
            # Predict UNet
            out_unet = unet(norm_img)
            if isinstance(out_unet, dict):
                out_unet = out_unet['main_output']
            p_unet = F.softmax(out_unet, dim=1)
            
            # Predict FCN
            out_fcn = fcn(norm_img)
            if isinstance(out_fcn, dict):
                out_fcn = out_fcn['main_output']
            p_fcn = F.softmax(out_fcn, dim=1)
            
            # Resize probability maps to original size
            p_dl_orig = F.interpolate(p_dl, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0).cpu().numpy()
            p_unet_orig = F.interpolate(p_unet, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0).cpu().numpy()
            p_fcn_orig = F.interpolate(p_fcn, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0).cpu().numpy()
            
            # Hybrid Blend
            fused_probs = (p_dl_orig * w_dl + p_unet_orig * w_unet + p_fcn_orig * w_fcn) / total_w
            pred_hybrid = torch.tensor(np.argmax(fused_probs, axis=0), device=device)
            
            for c in range(10):
                pred_c = (pred_hybrid == c) & valid_mask
                tgt_c = (target_safe == c) & valid_mask
                intersection_hybrid[c] += torch.sum(pred_c & tgt_c).item()
                union_hybrid[c] += torch.sum(pred_c | tgt_c).item()
                
            # Meta Learner Blend
            if meta_layer is not None:
                # Stack to [30, H, W]
                stacked = np.concatenate([p_unet_orig, p_fcn_orig, p_dl_orig], axis=0) # UNet, FCN, DeepLab
                stacked_tensor = torch.tensor(stacked[None, ...], dtype=torch.float32, device=device)
                meta_logits = meta_layer(stacked_tensor)
                meta_probs = F.softmax(meta_logits, dim=1).squeeze(0).cpu().numpy()
                pred_meta = torch.tensor(np.argmax(meta_probs, axis=0), device=device)
                
                for c in range(10):
                    pred_c = (pred_meta == c) & valid_mask
                    tgt_c = (target_safe == c) & valid_mask
                    intersection_meta[c] += torch.sum(pred_c & tgt_c).item()
                    union_meta[c] += torch.sum(pred_c | tgt_c).item()
                    
    dice_hybrid = (2. * intersection_hybrid) / (union_hybrid + intersection_hybrid + 1e-6)
    print(f"Hybrid Weighted Ensemble | Class 1-9 Avg Dice: {np.mean(dice_hybrid[1:]):.4f}")
    
    if meta_layer is not None:
        dice_meta = (2. * intersection_meta) / (union_meta + intersection_meta + 1e-6)
        print(f"Meta-Learner Stacked Ensemble | Class 1-9 Avg Dice: {np.mean(dice_meta[1:]):.4f}")
