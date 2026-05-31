import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import glob
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
from scipy.ndimage import distance_transform_edt, gaussian_filter

from synergistic_model import FloodNetSynergisticNet
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask

def neighbor_fill_cleanup(pred_labels: np.ndarray, min_area=100) -> np.ndarray:
    if min_area <= 0:
        return pred_labels
    
    noise_mask = np.zeros(pred_labels.shape, dtype=bool)
    for class_id in range(1, 10):
        class_mask = (pred_labels == class_id).astype(np.uint8)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for i in range(1, num_components):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                noise_mask[labels == i] = True
                
    if not np.any(noise_mask):
        return pred_labels
        
    indices = distance_transform_edt(noise_mask, return_distances=False, return_indices=True)
    clean_labels = pred_labels[indices[0], indices[1]]
    return clean_labels

def evaluate_config(w_main, w_unet, w_deeplab, w_fcn, tta_modes, sigma, min_area, thresh=0.99):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = FloodNetSynergisticNet(num_classes=10).to(device)
    weights_path = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=True)
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
        model.load_state_dict(state)
    model.eval()
    
    # Data
    tr_img = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    tr_msk = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    np.random.seed(42)
    # Using 30 images for speed
    indices = np.random.choice(len(tr_img), 150, replace=False)[:30]
    
    row_dices = []
    
    with torch.no_grad():
        for idx in indices:
            img_bgr = cv2.imread(tr_img[idx])
            orig_h, orig_w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            
            msk_rgb = cv2.cvtColor(cv2.imread(tr_msk[idx]), cv2.COLOR_BGR2RGB)
            msk_resized = cv2.resize(msk_rgb, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            gt_label = rgb_to_mask(msk_resized, id2color, 10)
            
            img_tensor = resized_img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            # Predict with TTA
            accum_probs = np.zeros((10, DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH), dtype=np.float32)
            weights_sum = 0.0
            
            for tta_mode in tta_modes:
                # 1. Augment
                if tta_mode == "none":
                    inp = img_tensor
                elif tta_mode == "hflip":
                    inp = torch.flip(img_tensor, dims=[3])
                elif tta_mode == "vflip":
                    inp = torch.flip(img_tensor, dims=[2])
                elif tta_mode == "rot180":
                    inp = torch.rot90(img_tensor, k=2, dims=[2, 3])
                else:
                    continue
                
                # 2. Forward
                out = model(inp)
                
                p_main = F.softmax(out['main_output'], dim=1).cpu().squeeze(0).numpy()
                p_unet = F.softmax(out['unet_output'], dim=1).cpu().squeeze(0).numpy()
                p_deeplab = F.softmax(out['deeplab_output'], dim=1).cpu().squeeze(0).numpy()
                p_fcn = F.softmax(out['fcn_output'], dim=1).cpu().squeeze(0).numpy()
                
                # 3. Blend outputs
                p_blend = w_main * p_main + w_unet * p_unet + w_deeplab * p_deeplab + w_fcn * p_fcn
                p_blend /= (w_main + w_unet + w_deeplab + w_fcn)
                
                # 4. Invert augmentation
                if tta_mode == "hflip":
                    p_blend = np.flip(p_blend, axis=2)
                elif tta_mode == "vflip":
                    p_blend = np.flip(p_blend, axis=1)
                elif tta_mode == "rot180":
                    p_blend = np.rot90(p_blend, k=-2, axes=(1, 2))
                
                # 5. Accumulate confidence-weighted
                conf = np.max(p_blend, axis=0, keepdims=True)
                accum_probs += p_blend * conf
                weights_sum += conf
                
            fused_probs = accum_probs / (weights_sum + 1e-6)
            
            # Resize
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(fused_probs[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # 6. Gaussian smoothing
            if sigma > 0:
                for c in range(10):
                    probs_resized[c] = gaussian_filter(probs_resized[c], sigma=sigma)
                probs_resized /= np.sum(probs_resized, axis=0, keepdims=True) + 1e-8
            
            # Class 0 Threshold
            pred_labels = np.argmax(probs_resized, axis=0).astype(np.uint8)
            c0_mask = (pred_labels == 0)
            low_conf_mask = c0_mask & (probs_resized[0] < thresh)
            if np.any(low_conf_mask):
                fallback = np.argmax(probs_resized[1:], axis=0) + 1
                pred_labels[low_conf_mask] = fallback[low_conf_mask].astype(np.uint8)
                
            # 7. Morphological Cleanup
            pred_labels = neighbor_fill_cleanup(pred_labels, min_area=min_area)
            
            # Evaluate Dice
            valid_mask = (gt_label != 255)
            img_dices = []
            for c in range(10):
                pred_c = (pred_labels == c) & valid_mask
                gt_c = (gt_label == c) & valid_mask
                
                intersection = np.sum(pred_c & gt_c)
                if np.sum(gt_c) == 0:
                    dice = 1.0 if np.sum(pred_c) == 0 else 0.0
                else:
                    dice = (2. * intersection) / (np.sum(pred_c) + np.sum(gt_c) + 1e-6)
                img_dices.append(dice)
            row_dices.append(np.mean(img_dices))
            
    return np.mean(row_dices)

def main():
    print("Running validation grid search to evaluate leveraging 91.093% model strategies...")
    
    # 1. Base reference (1.0 main-only, H-Flip TTA, no smoothing, neighbor fill=100)
    print("\n--- Baseline Reference (H-Flip TTA, No Blending, No Smoothing, Area=100) ---")
    score_ref = evaluate_config(1.0, 0.0, 0.0, 0.0, ["none", "hflip"], 0.0, 100)
    print(f"Ref Score: {score_ref:.5f}")
    
    # 2. Sweep Blending Weights (Main + Decoder outputs)
    print("\n--- Sweeping Blending Configurations ---")
    blends = [
        ("0.8 Main + 0.1 UNet + 0.1 DeepLab", 0.8, 0.1, 0.1, 0.0),
        ("0.6 Main + 0.2 UNet + 0.2 DeepLab", 0.6, 0.2, 0.2, 0.0),
        ("0.5 Main + 0.2 UNet + 0.2 DeepLab + 0.1 FCN", 0.5, 0.2, 0.2, 0.1),
        ("0.4 Main + 0.3 UNet + 0.3 DeepLab", 0.4, 0.3, 0.3, 0.0),
        ("0.33 Main + 0.33 UNet + 0.33 DeepLab", 0.33, 0.33, 0.33, 0.0)
    ]
    for desc, wm, wu, wd, wf in blends:
        score = evaluate_config(wm, wu, wd, wf, ["none", "hflip"], 0.0, 100)
        print(f"Config: {desc:45s} | Score: {score:.5f}")
        
    # 3. Sweep TTA Modes
    print("\n--- Sweeping TTA Modes (using 1.0 Main-only, no smoothing, area=100) ---")
    tta_options = [
        ("None (No TTA)", ["none"]),
        ("H-Flip TTA", ["none", "hflip"]),
        ("H-Flip + V-Flip TTA", ["none", "hflip", "vflip"]),
        ("H-Flip + V-Flip + Rot180 TTA", ["none", "hflip", "vflip", "rot180"])
    ]
    for desc, modes in tta_options:
        score = evaluate_config(1.0, 0.0, 0.0, 0.0, modes, 0.0, 100)
        print(f"Config: {desc:45s} | Score: {score:.5f}")
        
    # 4. Sweep Gaussian Smoothing Sigma
    print("\n--- Sweeping Gaussian Smoothing Sigma (using H-Flip TTA, no blending, area=100) ---")
    sigmas = [0.0, 0.05, 0.1, 0.2, 0.3]
    for s in sigmas:
        score = evaluate_config(1.0, 0.0, 0.0, 0.0, ["none", "hflip"], s, 100)
        print(f"Sigma: {s:<5.2f} | Score: {score:.5f}")

if __name__ == '__main__':
    main()
