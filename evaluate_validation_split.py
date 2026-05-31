import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
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
import gc

from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig, id2color, rgb_to_mask
from unet_version import StandardUNet
from fcn_version import ResNet50FCN

LOW_RES_EVAL = True # Set to True for fast, memory-safe validation. False for original high-res.

# Setup validation split (same as Keras)
image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))

train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    # img_tensor: (1, 3, H, W) normalized to [0, 1]
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
        # Apply saturation using TF_f on squeezed image
        squeezed = img_tensor.squeeze(0) # (3, H, W)
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
    # probs: (1, 10, H, W)
    if mode == "hflip":
        return TF_f.hflip(probs)
    elif mode == "vflip":
        return TF_f.vflip(probs)
    else:
        return probs

def normalize_imagenet(img_tensor):
    # img_tensor: (1, 3, H, W) in [0, 1]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (img_tensor - mean) / std

def get_deeplab_fused(model, img_tensor, mode_fusion=True, edge_sharpening=True):
    # img_tensor: normalized to ImageNet
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
    # probs: (10, H, W) numpy array
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
        for idx in tqdm(range(len(val_image_paths)), desc=f"Eval {model_type} [fusion={mode_fusion}, edge={edge_sharpening}, tta={use_tta}, smooth={use_smoothing}]"):
            img = cv2.cvtColor(cv2.imread(val_image_paths[idx]), cv2.COLOR_BGR2RGB)
            msk = cv2.cvtColor(cv2.imread(val_mask_paths[idx]), cv2.COLOR_BGR2RGB)
            
            orig_h, orig_w = img.shape[:2]
            
            if LOW_RES_EVAL:
                # Target mask resized to model resolution
                msk_resized = cv2.resize(msk, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
                label = rgb_to_mask(msk_resized, id2color, 10)
            else:
                # Target mask at original shape
                label = rgb_to_mask(msk, id2color, 10)
                
            target = torch.tensor(label, dtype=torch.long, device=device) # (H, W) or (480, 640)
            
            # Base Image in [0, 1] at model input size (480, 640)
            img_resized = cv2.resize(img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_01 = torch.tensor(img_resized.transpose(2, 0, 1)[None, ...], dtype=torch.float32, device=device) / 255.0 # (1, 3, H, W)
            
            # TTA prediction loop
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
                
                # Invert spatial predictions
                restored_probs = invert_tta_mode(pred_probs, mode) # (1, 10, H, W)
                
                if use_tta:
                    conf = restored_probs.max(dim=1, keepdim=True)[0] # (1, 1, H, W)
                    preds_all.append(restored_probs * conf)
                    confs_all.append(conf)
                else:
                    preds_all.append(restored_probs)
                    
            if use_tta:
                combined_probs = sum(preds_all) / (sum(confs_all) + 1e-6)
            else:
                combined_probs = preds_all[0]
                
            if LOW_RES_EVAL:
                # Keep probabilities at low resolution
                combined_probs = combined_probs.squeeze(0) # (10, 480, 640)
            else:
                # Upsample probabilities back to original shape
                combined_probs = F.interpolate(combined_probs, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0) # (10, orig_h, orig_w)
                
            probs_np = combined_probs.cpu().numpy()
            
            # Apply Gaussian smoothing if active
            if use_smoothing:
                probs_np = crf_like_smoothing(probs_np, sigma=0.1)
                
            pred_labels = torch.tensor(np.argmax(probs_np, axis=0), device=device) # (H, W) or (orig_h, orig_w)
            
            valid_mask = (target != 255)
            target_safe = torch.where(target == 255, torch.zeros_like(target), target)
            
            for c in range(10):
                pred_c = (pred_labels == c) & valid_mask
                tgt_c = (target_safe == c) & valid_mask
                intersection[c] += torch.sum(pred_c & tgt_c)
                union[c] += torch.sum(pred_c | tgt_c)
                
            # Free memory explicitly inside loop
            del img, msk, img_resized, img_01, preds_all, confs_all, combined_probs, probs_np, pred_labels, target, valid_mask, target_safe
            
    dice = (2. * intersection) / (union + intersection + 1e-6)
    
    # Cleanup after function
    gc.collect()
    torch.cuda.empty_cache()
    
    return dice.cpu().numpy()

if __name__ == '__main__':
    print("Loading models...")
    deeplab = CustomDeepLabV3Plus(num_classes=10).to(device)
    deeplab = load_weights_custom(deeplab, "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt", device)
    
    unet = StandardUNet(num_classes=10).to(device)
    unet = load_weights_custom(unet, "model_checkpoint/FloodNet_UNet/best_unet_weights.pt", device)
    
    fcn = ResNet50FCN(num_classes=10).to(device)
    fcn = load_weights_custom(fcn, "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt", device)
    
    print(f"\nEvaluating on full validation split ({len(val_image_paths)} images)...")
    if LOW_RES_EVAL:
        print("Using LOW_RES_EVAL = True (evaluating at model resolution 480x640 for speed and safety).")
    else:
        print("Using LOW_RES_EVAL = False (evaluating at native high resolution 3000x4000).")
        
    # 1. DeepLab Baseline
    dice_dl = evaluate_configuration(deeplab, 'dl', mode_fusion=False, edge_sharpening=False, use_tta=False, use_smoothing=False)
    print(f"DeepLab Baseline | Class 1-9 Avg Dice: {np.mean(dice_dl[1:]):.4f}")
    
    # 2. UNet Baseline
    dice_unet = evaluate_configuration(unet, 'unet', use_tta=False, use_smoothing=False)
    print(f"UNet Baseline | Class 1-9 Avg Dice: {np.mean(dice_unet[1:]):.4f}")
    
    # 3. FCN Baseline
    dice_fcn = evaluate_configuration(fcn, 'fcn', use_tta=False, use_smoothing=False)
    print(f"FCN Baseline | Class 1-9 Avg Dice: {np.mean(dice_fcn[1:]):.4f}")
    
    # 4. Ensemble Evaluation
    print("\nEvaluating Ensembles (no TTA, no DeepLab multi-scale fusion)...")
    w_dl = np.array([1.1350, 0.0000, 0.0000, 0.0000, 0.0000, 2.5432, 0.0000, 0.0000, 0.9120, 0.0000], dtype=np.float32).reshape(10, 1, 1)
    w_unet = np.array([0.0000, 0.9893, 2.2479, 0.7506, 1.2237, 1.8659, 2.7888, 1.9066, 0.9812, 1.4061], dtype=np.float32).reshape(10, 1, 1)
    w_fcn = np.array([1.8123, 0.8671, 1.3632, 1.2473, 0.9071, 1.0935, 0.8761, 0.6374, 1.0507, 0.9598], dtype=np.float32).reshape(10, 1, 1)
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
        for idx in tqdm(range(len(val_image_paths)), desc="Running Ensembles"):
            img = cv2.cvtColor(cv2.imread(val_image_paths[idx]), cv2.COLOR_BGR2RGB)
            msk = cv2.cvtColor(cv2.imread(val_mask_paths[idx]), cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
            
            if LOW_RES_EVAL:
                msk_resized = cv2.resize(msk, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
                label = rgb_to_mask(msk_resized, id2color, 10)
            else:
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
            
            if LOW_RES_EVAL:
                # Squeeze to CPU numpy directly at 480x640
                p_dl_orig = p_dl.squeeze(0).cpu().numpy()
                p_unet_orig = p_unet.squeeze(0).cpu().numpy()
                p_fcn_orig = p_fcn.squeeze(0).cpu().numpy()
            else:
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
                if LOW_RES_EVAL:
                    # Stack on CPU at 480x640
                    stacked = np.concatenate([p_unet_orig, p_fcn_orig, p_dl_orig], axis=0) # UNet, FCN, DeepLab
                    stacked_tensor = torch.tensor(stacked[None, ...], dtype=torch.float32, device=device)
                    meta_logits = meta_layer(stacked_tensor)
                    meta_probs = F.softmax(meta_logits, dim=1).squeeze(0).cpu().numpy()
                    pred_meta = torch.tensor(np.argmax(meta_probs, axis=0), device=device)
                else:
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
            
            # Free iteration memory
            del img, msk, target, valid_mask, target_safe, img_resized, img_01, norm_img
            del out_dl, p_dl, out_unet, p_unet, out_fcn, p_fcn
            del p_dl_orig, p_unet_orig, p_fcn_orig, fused_probs, pred_hybrid
            if meta_layer is not None:
                del stacked, stacked_tensor, meta_logits, meta_probs, pred_meta
                
    dice_hybrid = (2. * intersection_hybrid) / (union_hybrid + intersection_hybrid + 1e-6)
    print(f"Hybrid Weighted Ensemble | Class 1-9 Avg Dice: {np.mean(dice_hybrid[1:]):.4f}")
    
    if meta_layer is not None:
        dice_meta = (2. * intersection_meta) / (union_meta + intersection_meta + 1e-6)
        print(f"Meta-Learner Stacked Ensemble | Class 1-9 Avg Dice: {np.mean(dice_meta[1:]):.4f}")

    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
