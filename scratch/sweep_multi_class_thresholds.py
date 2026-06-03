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

def neighbor_fill_cleanup(pred_labels: np.ndarray, min_area=8) -> np.ndarray:
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

def compute_image_class_dices_bincount(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = (target != 255)
    pred_valid = pred[valid]
    target_valid = target[valid]
    
    if len(target_valid) == 0:
        return np.ones(10)
        
    pred_counts = np.bincount(pred_valid, minlength=10)
    tgt_counts = np.bincount(target_valid, minlength=10)
    
    matches = target_valid[target_valid == pred_valid]
    intersections = np.bincount(matches, minlength=10)
    
    dices = np.zeros(10)
    for c in range(10):
        sp = pred_counts[c]
        st = tgt_counts[c]
        if sp == 0 and st == 0:
            dices[c] = 1.0
        elif sp == 0 or st == 0:
            dices[c] = 0.0
        else:
            dices[c] = (2.0 * intersections[c]) / (sp + st)
    return dices

def apply_multiclass_thresholding_fast(pred: np.ndarray, probs: np.ndarray, thresh_dict: dict, fallback: np.ndarray) -> np.ndarray:
    pred_clean = pred.copy()
    for c, t in thresh_dict.items():
        if t > 0:
            mask = (pred_clean == c) & (probs[c] < t)
            pred_clean[mask] = fallback[mask]
    return pred_clean

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    _, val_img_paths, _, val_msk_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    val_img_paths = val_img_paths[:80]
    val_msk_paths = val_msk_paths[:80]
    print(f"Found {len(val_img_paths)} validation target images.")
    
    # 1. Precompute Synergistic Model Predictions at 480x640
    print("\n--- Phase 1: Precomputing Synergistic (Low-Res) ---")
    syn_model = FloodNetSynergisticNet(num_classes=10).to(device)
    syn_weights = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    if os.path.exists(syn_weights):
        state = torch.load(syn_weights, map_location=device, weights_only=True)
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
        syn_model.load_state_dict(state)
    syn_model.eval()
    
    syn_probs_list = []
    targets_list = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_img_paths)), desc="Synergistic"):
            img_bgr = cv2.imread(val_img_paths[idx])
            
            msk = cv2.imread(val_msk_paths[idx])
            msk_rgb = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
            msk_resized = cv2.resize(msk_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            label = rgb_to_mask(msk_resized, id2color, 10)
            targets_list.append(label)
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            out_std = syn_model(img_tensor)
            p_main = F.softmax(out_std['main_output'], dim=1).cpu().squeeze(0).numpy()
            p_unet = F.softmax(out_std['unet_output'], dim=1).cpu().squeeze(0).numpy()
            p_dl = F.softmax(out_std['deeplab_output'], dim=1).cpu().squeeze(0).numpy()
            
            p_syn_blend = 0.4 * p_main + 0.3 * p_unet + 0.3 * p_dl
            p_syn_blend[0] = p_main[0]
            
            syn_probs_list.append(p_syn_blend)
            
    del syn_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Precompute Meta-Stacked Model Predictions at 480x640
    print("\n--- Phase 2: Precomputing Meta-Stacked (Low-Res) ---")
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
        for idx in tqdm(range(len(val_img_paths)), desc="Meta-Stacked"):
            img_bgr = cv2.imread(val_img_paths[idx])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            meta_logits = meta_model(img_tensor)
            meta_probs = F.softmax(meta_logits, dim=1).squeeze(0).cpu().numpy()
            meta_probs_list.append(meta_probs)
            
    del meta_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. Fine-tuned Joint Sweep over W_Syn, Thresh C3, and Area
    # Keep Thresh C0 = 0.90, Thresh C1 = 0.00 fixed
    print("\n--- Phase 3: Fine-tuned Joint Sweep ---")
    
    w_syn_list = [0.45, 0.50, 0.55, 0.60]
    t3_list = [0.00, 0.30, 0.50, 0.70]
    areas = [0, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    
    results = []
    
    for w_syn in tqdm(w_syn_list, desc="Weights Sweep"):
        w_meta = 1.0 - w_syn
        
        # Precompute fused probability maps and argmax predictions
        fused_probs_list = [w_syn * sp + w_meta * mp for sp, mp in zip(syn_probs_list, meta_probs_list)]
        preds_list = [np.argmax(probs, axis=0) for probs in fused_probs_list]
        
        # Precompute the fallback lists
        # fallback_c03: suppress 0 and 3. fallback_c0: suppress 0
        fallback_c0 = [np.argmax(probs[1:], axis=0) + 1 for probs in fused_probs_list]
        
        non_c03 = [1, 2, 4, 5, 6, 7, 8, 9]
        fallback_c03 = [np.array(non_c03)[np.argmax(probs[non_c03], axis=0)] for probs in fused_probs_list]
        
        for t3 in t3_list:
            thresh_dict = {0: 0.90, 1: 0.00, 3: t3}
            fallback_list = fallback_c03 if t3 > 0 else fallback_c0
            
            # Precompute thresholded predictions
            thresh_preds = [apply_multiclass_thresholding_fast(pred, probs, thresh_dict, fallback) for pred, probs, fallback in zip(preds_list, fused_probs_list, fallback_list)]
            
            for area in areas:
                dices_sum = np.zeros(10)
                for pred_labels, target in zip(thresh_preds, targets_list):
                    pred_clean = pred_labels.copy()
                    if area > 0:
                        pred_clean = neighbor_fill_cleanup(pred_clean.astype(np.uint8), min_area=area)
                        
                    dices_sum += compute_image_class_dices_bincount(pred_clean, target)
                    
                class_means = dices_sum / len(targets_list)
                overall_mean = np.mean(class_means)
                
                results.append({
                    'w_syn': w_syn,
                    't3': t3,
                    'area': area,
                    'mean_dice': overall_mean,
                    'class_0': class_means[0],
                    'class_1': class_means[1],
                    'class_3': class_means[3],
                })
                
    results = sorted(results, key=lambda x: x['mean_dice'], reverse=True)
    
    print("\n" + "="*80)
    print("TOP 30 FINE-TUNED CONFIGURATIONS")
    print("="*80)
    for idx, r in enumerate(results[:30]):
        print(f"{idx+1:02d}. W_Syn: {r['w_syn']:.2f} | Thresh C3: {r['t3']:.2f} | Area: {r['area']:3d} || Mean Dice: {r['mean_dice']:.5f} | C0: {r['class_0']:.4f} | C1: {r['class_1']:.4f} | C3: {r['class_3']:.4f}")
    print("="*80)

if __name__ == '__main__':
    main()
