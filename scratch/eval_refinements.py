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
from multiprocessing import Pool
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
    return (2.0 * intersection) / (sum_pred + sum_tgt + 1e-8)

def neighbor_fill_cleanup_class_specific(pred_labels: np.ndarray, class_areas: list) -> np.ndarray:
    noise_mask = np.zeros(pred_labels.shape, dtype=bool)
    for class_id in range(10):
        area = class_areas[class_id]
        if area <= 0:
            continue
        class_mask = (pred_labels == class_id).astype(np.uint8)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for i in range(1, num_components):
            if stats[i, cv2.CC_STAT_AREA] < area:
                noise_mask[labels == i] = True
                
    if not np.any(noise_mask):
        return pred_labels
        
    indices = distance_transform_edt(noise_mask, return_distances=False, return_indices=True)
    clean_labels = pred_labels[indices[0], indices[1]]
    return clean_labels

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load optimal config
    config_path = "model_checkpoint/ensemble_kaggle_config.pt"
    if not os.path.exists(config_path):
        print(f"Error: config not found at {config_path}!")
        return
        
    config = torch.load(config_path)
    best_w = config['best_w']
    best_t = config['best_t']
    best_areas_low = config.get('best_areas', [config.get('best_area', 96)] * 10)
    
    print("\nLoaded Optimal Hyperparameters:")
    print("  - Blending weights (w_syn):", [round(w, 4) for w in best_w])
    print("  - Probability thresholds:", [round(t, 4) for t in best_t])
    print("  - Morphological areas (low-res):", best_areas_low)
    
    # Validation split
    image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    _, val_image_paths, _, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # Evaluate on a slice of 200 validation images in original high-resolution
    num_eval = 200
    val_image_paths = val_image_paths[:num_eval]
    val_mask_paths = val_mask_paths[:num_eval]
    print(f"Evaluating on {num_eval} validation images at original high resolution...")
    
    print("Loading Models...")
    model_syn = FloodNetSynergisticNet(num_classes=10).to(device)
    syn_weights = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    state = torch.load(syn_weights, map_location=device, weights_only=True)
    state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
    model_syn.load_state_dict(state)
    model_syn.eval()
    
    model_meta = FloodNetCompetitiveModel(num_classes=10).to(device)
    model_meta.load_checkpoints(
        unet_path="model_checkpoint/FloodNet_UNet/best_unet_weights.pt",
        fcn_path="model_checkpoint/FloodNet_FCN/best_fcn_weights.pt",
        deeplab_path="model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt",
        meta_path="model_checkpoint/FloodNet_Meta/meta_layer_weights.pt",
        device=device
    )
    model_meta.eval()
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Store results
    dices_nearest = []
    dices_bilinear = []
    
    best_t_arr = np.array(best_t)
    best_w_arr = np.array(best_w).reshape(10, 1, 1)
    
    fallback_classes = [c for c in range(10) if best_t_arr[c] == 0.0]
    if len(fallback_classes) == 0:
        fallback_classes = [2]
        
    print("\nStarting evaluation sweep...")
    with torch.no_grad():
        for idx in tqdm(range(num_eval), desc="Evaluating"):
            img_bgr = cv2.imread(val_image_paths[idx])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            msk_rgb = cv2.cvtColor(cv2.imread(val_mask_paths[idx]), cv2.COLOR_BGR2RGB)
            
            orig_h, orig_w = img_rgb.shape[:2]
            
            # Ground truth (original resolution)
            gt_label = rgb_to_mask(msk_rgb, id2color, 10)
            valid_mask = (gt_label != 255)
            
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
            p_syn[:, 0] = p_syn_main[:, 0]
            p_syn_np = p_syn.cpu().squeeze(0).numpy()
            
            # Predict Meta Net
            meta_logits = model_meta(x)
            p_meta_np = F.softmax(meta_logits, dim=1).cpu().squeeze(0).numpy()
            
            # Blend at low-res
            probs_low = best_w_arr * p_syn_np + (1.0 - best_w_arr) * p_meta_np
            probs_low[0] = p_syn_np[0]
            
            # ----------------------------------------------------
            # Method 1: Nearest-Neighbor Mask Upsampling (Baseline)
            # ----------------------------------------------------
            pred_low = np.argmax(probs_low, axis=0)
            fallback_probs_low = probs_low[fallback_classes]
            fallback_idx_low = np.argmax(fallback_probs_low, axis=0)
            fallback_low = np.array(fallback_classes)[fallback_idx_low]
            
            for c in range(10):
                t = best_t_arr[c]
                if t > 0.0:
                    mask = (pred_low == c) & (probs_low[c] < t)
                    pred_low[mask] = fallback_low[mask]
            
            # Cleanup at low resolution
            pred_low_cleaned = neighbor_fill_cleanup_class_specific(pred_low, best_areas_low)
            
            # Nearest neighbor resize to high resolution
            pred_hr_nearest = cv2.resize(pred_low_cleaned, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # Eval Method 1
            img_dices_n = []
            for c in range(10):
                pred_c = (pred_hr_nearest == c) & valid_mask
                tgt_c = (gt_label == c) & valid_mask
                img_dices_n.append(row_wise_dice(pred_c, tgt_c))
            dices_nearest.append(np.mean(img_dices_n))
            
            # ----------------------------------------------------
            # Method 2: Bilinear Probability Upsampling
            # ----------------------------------------------------
            # Upsample probability maps to high resolution
            # PyTorch F.interpolate is extremely fast on GPU
            probs_low_t = torch.tensor(probs_low[None, ...], dtype=torch.float32, device=device)
            probs_hr_t = F.interpolate(probs_low_t, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0)
            probs_hr = probs_hr_t.cpu().numpy()
            
            pred_hr_bilinear = np.argmax(probs_hr, axis=0)
            fallback_probs_hr = probs_hr[fallback_classes]
            fallback_idx_hr = np.argmax(fallback_probs_hr, axis=0)
            fallback_hr = np.array(fallback_classes)[fallback_idx_hr]
            
            for c in range(10):
                t = best_t_arr[c]
                if t > 0.0:
                    mask = (pred_hr_bilinear == c) & (probs_hr[c] < t)
                    pred_hr_bilinear[mask] = fallback_hr[mask]
            
            # Class areas scale with the image scale (area ratio)
            scale_ratio = (orig_h / DatasetConfig.IMG_HEIGHT) * (orig_w / DatasetConfig.IMG_WIDTH)
            best_areas_hr = [int(a * scale_ratio) for a in best_areas_low]
            
            # Cleanup at high resolution
            pred_hr_bilinear_cleaned = neighbor_fill_cleanup_class_specific(pred_hr_bilinear, best_areas_hr)
            
            # Eval Method 2
            img_dices_b = []
            for c in range(10):
                pred_c = (pred_hr_bilinear_cleaned == c) & valid_mask
                tgt_c = (gt_label == c) & valid_mask
                img_dices_b.append(row_wise_dice(pred_c, tgt_c))
            dices_bilinear.append(np.mean(img_dices_b))
            
            # Clean up
            del x, out_syn, p_syn_main, p_syn_unet, p_syn_dl, p_syn, meta_logits, probs_low_t, probs_hr_t
            
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Method 1 (Nearest Label Resize):   {np.mean(dices_nearest):.6f}")
    print(f"Method 2 (Bilinear Probability):  {np.mean(dices_bilinear):.6f}")
    print(f"Absolute Dice Gain:               {np.mean(dices_bilinear) - np.mean(dices_nearest):+.6f}")
    print("====================================================")

if __name__ == '__main__':
    main()
