import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"
import glob
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import gc

from synergistic_model import FloodNetSynergisticNet
from competitive_model import FloodNetCompetitiveModel
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask
from scratch.eval_advanced_tta import neighbor_fill_cleanup

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load models
    print("Loading Synergistic Model...")
    model_syn = FloodNetSynergisticNet(num_classes=10).to(device)
    syn_weights = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    state = torch.load(syn_weights, map_location=device, weights_only=True)
    state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
    model_syn.load_state_dict(state)
    model_syn.eval()
    
    print("Loading Meta-Stacked Model...")
    model_meta = FloodNetCompetitiveModel(num_classes=10).to(device)
    model_meta.load_checkpoints(
        unet_path="model_checkpoint/FloodNet_UNet/best_unet_weights.pt",
        fcn_path="model_checkpoint/FloodNet_FCN/best_fcn_weights.pt",
        deeplab_path="model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt",
        meta_path="model_checkpoint/FloodNet_Meta/meta_layer_weights.pt",
        device=device
    )
    model_meta.eval()
    
    # Disable cudnn benchmark for stability on ROCm/AMD
    torch.backends.cudnn.benchmark = False
    
    # 2. Get validation images
    image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    _, val_image_paths, _, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    num_val = len(val_image_paths)
    print(f"Total Validation Images: {num_val}")
    
    # Allocate tensors on GPU
    # size: (num_val, 10, 480, 640) float16
    P_syn = torch.zeros((num_val, 10, 480, 640), dtype=torch.float16, device=device)
    P_meta = torch.zeros((num_val, 10, 480, 640), dtype=torch.float16, device=device)
    Targets = torch.zeros((num_val, 480, 640), dtype=torch.uint8, device=device)
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    print("\nPrecomputing predictions in memory (float16 on GPU)...")
    with torch.no_grad():
        for idx in tqdm(range(num_val), desc="Inference"):
            img_bgr = cv2.imread(val_image_paths[idx])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            msk_rgb = cv2.cvtColor(cv2.imread(val_mask_paths[idx]), cv2.COLOR_BGR2RGB)
            
            # Ground truth
            msk_resized = cv2.resize(msk_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            gt_label = rgb_to_mask(msk_resized, id2color, 10)
            Targets[idx] = torch.tensor(gt_label, dtype=torch.uint8, device=device)
            
            # Input preprocessing
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
            x = torch.tensor(img_norm.transpose(2, 0, 1)[None, ...], dtype=torch.float32).to(device)
            
            # Predict Synergistic Net
            out_syn = model_syn(x)
            p_syn_main = F.softmax(out_syn['main_output'], dim=1)
            p_syn_unet = F.softmax(out_syn['unet_output'], dim=1)
            p_syn_dl = F.softmax(out_syn['deeplab_output'], dim=1)
            p_syn = 0.4 * p_syn_main + 0.3 * p_syn_unet + 0.3 * p_syn_dl
            p_syn[:, 0] = p_syn_main[:, 0] # Keep background class unscaled
            del p_syn_main, p_syn_unet, p_syn_dl, out_syn
            
            # Predict Meta Net (Standard single-pass, no TTA)
            meta_logits = model_meta(x)
            p_meta_pred = F.softmax(meta_logits, dim=1)
            
            # Save to float16 tensors
            P_syn[idx] = p_syn.squeeze(0).to(torch.float16)
            P_meta[idx] = p_meta_pred.squeeze(0).to(torch.float16)
            
            # Free temporary GPU variables
            del x, p_syn, p_meta_pred
            
            
    # Delete models to free VRAM
    del model_syn, model_meta
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nPrecomputation complete!")
    print(f"GPU Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"GPU Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    
    # Helper evaluation function using batching to save VRAM
    def evaluate_config_torch(w_syn_arr, thresh_arr):
        # w_syn_arr: shape (10,)
        # thresh_arr: shape (10,)
        w_syn_t = torch.tensor(w_syn_arr, dtype=torch.float16, device=device).view(1, 10, 1, 1)
        thresh_t = torch.tensor(thresh_arr, dtype=torch.float16, device=device)
        
        # Identify fallback classes (those with threshold 0.0)
        fallback_classes = [c for c in range(10) if thresh_arr[c] == 0.0]
        if len(fallback_classes) == 0:
            fallback_classes = [2] # default fallback to class 2 (building)
            
        fallback_classes_t = torch.tensor(fallback_classes, device=device)
        
        class_intersections = torch.zeros(10, dtype=torch.float32, device=device)
        class_unions = torch.zeros(10, dtype=torch.float32, device=device)
        
        batch_size = 32
        num_images = P_syn.shape[0]
        
        for i in range(0, num_images, batch_size):
            end_idx = min(i + batch_size, num_images)
            p_syn_b = P_syn[i:end_idx]
            p_meta_b = P_meta[i:end_idx]
            targets_b = Targets[i:end_idx]
            
            # Blend probabilities
            probs_b = w_syn_t * p_syn_b + (1.0 - w_syn_t) * p_meta_b
            probs_b[:, 0] = p_syn_b[:, 0] # Class 0 unscaled
            
            # Multiclass thresholding fallback logic
            pred_b = torch.argmax(probs_b, dim=1)
            
            fallback_probs_b = probs_b[:, fallback_classes, :, :]
            fallback_idx_b = torch.argmax(fallback_probs_b, dim=1)
            fallback_b = fallback_classes_t[fallback_idx_b]
            
            for c in range(10):
                t = thresh_t[c]
                if t > 0.0:
                    mask = (pred_b == c) & (probs_b[:, c] < t)
                    pred_b = torch.where(mask, fallback_b, pred_b)
                    
            valid_mask = (targets_b != 255)
            for c in range(1, 10):
                pred_c = (pred_b == c) & valid_mask
                gt_c = (targets_b == c) & valid_mask
                
                class_intersections[c] += torch.sum(pred_c & gt_c)
                class_unions[c] += torch.sum(pred_c) + torch.sum(gt_c)
            
        dices = (2.0 * class_intersections[1:]) / (class_unions[1:] + 1e-6)
        macro_dice = torch.mean(dices).item()
        
        return macro_dice
        
    # Baseline score
    baseline_w = np.ones(10) * 0.50
    baseline_t = np.zeros(10)
    baseline_t[0] = 0.95
    baseline_t[1] = 0.50
    baseline_t[3] = 0.50
    
    baseline_dice = evaluate_config_torch(baseline_w, baseline_t)
    print(f"\nBaseline Validation Dice (w_syn=0.50, c0=0.95, c1=0.50, c3=0.50): {baseline_dice:.6f}")
    
    # 3. Powell Optimization of Weights and Thresholds
    print("\nStarting Powell optimization of class-specific blending weights and thresholds...")
    
    # Parameter vector x: first 10 element are W_syn, next 10 elements are Thresh
    def loss_function(x):
        w_syn = np.clip(x[0:10], 0.0, 1.0)
        thresh = np.clip(x[10:20], 0.0, 1.0)
        
        # Constrain thresholds to be 0 if they are below 0.02 to allow unconstrained fallback set
        thresh[thresh < 0.02] = 0.0
        
        # Maximize Dice -> minimize negative Dice
        score = evaluate_config_torch(w_syn, thresh)
        return -score
        
    initial_x = np.concatenate([
        np.ones(10) * 0.50, # w_syn initial
        baseline_t          # thresh initial
    ])
    
    # Run Powell method
    res = minimize(loss_function, initial_x, method='Powell', options={'maxiter': 10, 'disp': True})
    
    best_x = res.x
    best_w = np.clip(best_x[0:10], 0.0, 1.0)
    best_t = np.clip(best_x[10:20], 0.0, 1.0)
    best_t[best_t < 0.02] = 0.0
    
    optimized_dice = -res.fun
    print("\n================= OPTIMIZATION RESULTS =================")
    print(f"Optimal Hyperparameters:")
    print("Class-specific Synergistic Weights (w_syn):")
    for c in range(10):
        print(f"  Class {c}: {best_w[c]:.4f} (Meta Weight: {1.0 - best_w[c]:.4f})")
        
    print("\nClass-specific Thresholds (thresh):")
    for c in range(10):
        if best_t[c] > 0.0:
            print(f"  Class {c}: {best_t[c]:.4f}")
        else:
            print(f"  Class {c}: 0.0000 (No suppression / Fallback class)")
            
    print(f"\nBest Validation Dice: {optimized_dice:.6f} (Delta: {optimized_dice - baseline_dice:+.6f})")
    
    # 4. Sweep min_area Morphological Cleanup on CPU
    print("\nSweeping min_area post-processing parameter on CPU...")
    # Convert best predictions to CPU to run morphological cleanup
    w_syn_t = torch.tensor(best_w, dtype=torch.float16, device=device).view(1, 10, 1, 1)
    thresh_t = torch.tensor(best_t, dtype=torch.float16, device=device)
    
    fallback_classes = [c for c in range(10) if best_t[c] == 0.0]
    if len(fallback_classes) == 0:
        fallback_classes = [2]
    fallback_classes_t = torch.tensor(fallback_classes, device=device)
    
    numpy_preds = np.zeros((num_val, 480, 640), dtype=np.uint8)
    numpy_targets = Targets.cpu().numpy()
    
    batch_size = 32
    num_images = P_syn.shape[0]
    
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            end_idx = min(i + batch_size, num_images)
            p_syn_b = P_syn[i:end_idx]
            p_meta_b = P_meta[i:end_idx]
            
            probs_b = w_syn_t * p_syn_b + (1.0 - w_syn_t) * p_meta_b
            probs_b[:, 0] = p_syn_b[:, 0]
            
            pred_b = torch.argmax(probs_b, dim=1)
            
            fallback_probs_b = probs_b[:, fallback_classes, :, :]
            fallback_idx_b = torch.argmax(fallback_probs_b, dim=1)
            fallback_b = fallback_classes_t[fallback_idx_b]
            
            for c in range(10):
                t = thresh_t[c]
                if t > 0.0:
                    mask = (pred_b == c) & (probs_b[:, c] < t)
                    pred_b = torch.where(mask, fallback_b, pred_b)
            
            numpy_preds[i:end_idx] = pred_b.cpu().numpy().astype(np.uint8)
        
    min_areas = [0, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256]
    best_area = 0
    best_area_dice = 0.0
    
    for area in min_areas:
        class_intersections = np.zeros(10)
        class_unions = np.zeros(10)
        
        for idx in range(num_val):
            pred_cleaned = neighbor_fill_cleanup(numpy_preds[idx].astype(np.uint8), min_area=area)
            valid_mask = (numpy_targets[idx] != 255)
            
            for c in range(1, 10):
                pred_c = (pred_cleaned == c) & valid_mask
                gt_c = (numpy_targets[idx] == c) & valid_mask
                class_intersections[c] += np.sum(pred_c & gt_c)
                class_unions[c] += np.sum(pred_c) + np.sum(gt_c)
                
        dices = (2.0 * class_intersections[1:]) / (class_unions[1:] + 1e-6)
        area_dice = np.mean(dices)
        print(f"min_area = {area:<3d} -> Validation Dice: {area_dice:.6f}")
        
        if area_dice > best_area_dice:
            best_area_dice = area_dice
            best_area = area
            
    print(f"\nOptimal Morphological Area: {best_area} with Dice score: {best_area_dice:.6f}")
    
    # Save optimized parameters to configuration file
    config_dict = {
        'best_w': best_w.tolist(),
        'best_t': best_t.tolist(),
        'best_area': best_area,
        'baseline_dice': baseline_dice,
        'optimized_dice': optimized_dice,
        'final_dice_with_area': best_area_dice
    }
    torch.save(config_dict, "model_checkpoint/ensemble_optimized_config.pt")
    print(f"\nConfiguration saved successfully to model_checkpoint/ensemble_optimized_config.pt")

if __name__ == '__main__':
    main()
