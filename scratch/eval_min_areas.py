import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.ndimage import distance_transform_edt
import gc

from synergistic_model import FloodNetSynergisticNet
from competitive_model import FloodNetCompetitiveModel
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask

def row_wise_dice(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    sum_pred = np.sum(pred_mask)
    sum_tgt = np.sum(target_mask)
    if sum_pred == 0 and sum_tgt == 0:
        return 1.0
    if sum_pred == 0 or sum_tgt == 0:
        return 0.0
    intersection = np.sum(pred_mask & target_mask)
    return (2.0 * intersection) / (sum_pred + sum_tgt)

def neighbor_fill_cleanup_all(pred_labels: np.ndarray, min_area=25, clean_class0=False) -> np.ndarray:
    if min_area <= 0:
        return pred_labels
    
    noise_mask = np.zeros(pred_labels.shape, dtype=bool)
    start_class = 0 if clean_class0 else 1
    for class_id in range(start_class, 10):
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Validation split
    image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    _, val_img_paths, _, val_msk_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # Evaluate on 100 validation images for robustness
    val_img_paths = val_img_paths[:100]
    val_msk_paths = val_msk_paths[:100]
    print(f"Found {len(val_img_paths)} validation target images.")
    
    # --- PHASE 1: PRECOMPUTE SYNERGISTIC MODEL PREDICTIONS ---
    print("\n--- Phase 1: Precomputing Synergistic Model Predictions ---")
    syn_model = FloodNetSynergisticNet(num_classes=10).to(device)
    syn_weights = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    if os.path.exists(syn_weights):
        print(f"Loading synergistic weights from {syn_weights}")
        state = torch.load(syn_weights, map_location=device, weights_only=True)
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
        syn_model.load_state_dict(state)
    syn_model.eval()
    
    syn_probs_list = []
    targets_list = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_img_paths))):
            img_bgr = cv2.imread(val_img_paths[idx])
            orig_h, orig_w = img_bgr.shape[:2]
            
            msk = cv2.imread(val_msk_paths[idx])
            msk_rgb = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
            label = rgb_to_mask(msk_rgb, id2color, 10)
            targets_list.append(label)
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            # Predict Synergistic Net with TTA (horizontal flip)
            img_flipped = torch.flip(img_tensor, dims=[3])
            out_std = syn_model(img_tensor)
            out_flip = syn_model(img_flipped)
            
            p_main_std = F.softmax(out_std['main_output'], dim=1)
            p_main_flip = F.softmax(out_flip['main_output'], dim=1)
            p_main = (p_main_std + torch.flip(p_main_flip, dims=[3])) * 0.5
            p_main_np = p_main.cpu().squeeze(0).numpy()
            
            p_unet_std = F.softmax(out_std['unet_output'], dim=1)
            p_unet_flip = F.softmax(out_flip['unet_output'], dim=1)
            p_unet = (p_unet_std + torch.flip(p_unet_flip, dims=[3])) * 0.5
            p_unet_np = p_unet.cpu().squeeze(0).numpy()
            
            p_dl_std = F.softmax(out_std['deeplab_output'], dim=1)
            p_dl_flip = F.softmax(out_flip['deeplab_output'], dim=1)
            p_dl = (p_dl_std + torch.flip(p_dl_flip, dims=[3])) * 0.5
            p_dl_np = p_dl.cpu().squeeze(0).numpy()
            
            p_syn_blend = 0.6 * p_main_np + 0.2 * p_unet_np + 0.2 * p_dl_np
            p_syn_blend[0] = p_main_np[0] # keep class 0 unscaled
            
            # Resize probability maps to original size
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(p_syn_blend[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            syn_probs_list.append(probs_resized)
            
    # Free memory
    del syn_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- PHASE 2: PRECOMPUTE META-STACKED MODEL PREDICTIONS ---
    print("\n--- Phase 2: Precomputing Meta-Stacked Model Predictions ---")
    meta_model = FloodNetCompetitiveModel(num_classes=10).to(device)
    meta_model.load_checkpoints(
        unet_path="model_checkpoint/FloodNet_UNet/best_unet_weights.pt",
        fcn_path="model_checkpoint/FloodNet_FCN/best_fcn_weights.pt",
        deeplab_path="model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt",
        meta_path="model_checkpoint/FloodNet_Meta/meta_layer_weights.pt",
        device=device
    )
    meta_model.eval()
    
    meta_probs_list = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_img_paths))):
            img_bgr = cv2.imread(val_img_paths[idx])
            orig_h, orig_w = img_bgr.shape[:2]
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            meta_logits = meta_model(img_tensor)
            meta_probs = F.softmax(meta_logits, dim=1).squeeze(0).cpu().numpy()
            
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(meta_probs[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            meta_probs_list.append(probs_resized)
            
    # Free memory
    del meta_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- PHASE 3: GRID SEARCH OVER BLENDING AND MIN_AREAS ---
    # Best weights from Stage 1: w_syn = 0.70, w_meta = 0.30
    w_syn = 0.70
    w_meta = 0.30
    
    print("\n--- Phase 3: Sweeping Morphological Cleanup Settings ---")
    min_areas = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100]
    thresh_c0 = 0.30
    
    # We will test both clean_class0 = False and True
    for clean_c0 in [False, True]:
        print(f"\nEvaluating with clean_class0 = {clean_c0}:")
        for area in min_areas:
            row_dices = {c: [] for c in range(10)}
            for syn_p, meta_p, target in zip(syn_probs_list, meta_probs_list, targets_list):
                probs = w_syn * syn_p + w_meta * meta_p
                
                # Apply Class 0 suppression
                pred_labels = np.argmax(probs, axis=0)
                class0_mask = (pred_labels == 0)
                low_conf_mask = class0_mask & (probs[0] < thresh_c0)
                if np.any(low_conf_mask):
                    fallback_labels = np.argmax(probs[1:], axis=0) + 1
                    pred_labels[low_conf_mask] = fallback_labels[low_conf_mask]
                    
                # Morphological cleanup
                if area > 0:
                    pred_labels = neighbor_fill_cleanup_all(pred_labels, min_area=area, clean_class0=clean_c0)
                    
                valid_mask = (target != 255)
                for c in range(10):
                    pred_c = (pred_labels == c) & valid_mask
                    tgt_c = (target == c) & valid_mask
                    row_dices[c].append(row_wise_dice(pred_c, tgt_c))
                    
            class_means = {c: np.mean(row_dices[c]) for c in range(10)}
            overall_mean = np.mean([class_means[c] for c in range(10)])
            print(f"  Min Area: {area:3d} | Mean Dice: {overall_mean:.5f} | Class 0 Dice: {class_means[0]:.5f} | Class 1 Dice: {class_means[1]:.5f}")

if __name__ == '__main__':
    main()
