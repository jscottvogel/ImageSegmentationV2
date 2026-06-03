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
import gc

from synergistic_model import FloodNetSynergisticNet
from competitive_model import FloodNetCompetitiveModel
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask
from scratch.eval_advanced_tta import neighbor_fill_cleanup

def row_wise_dice(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    sum_pred = np.sum(pred_mask)
    sum_tgt = np.sum(target_mask)
    if sum_pred == 0 and sum_tgt == 0:
        return 1.0
    if sum_pred == 0 or sum_tgt == 0:
        return 0.0
    intersection = np.sum(pred_mask & target_mask)
    return (2.0 * intersection) / (sum_pred + sum_tgt)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Validation split
    image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    _, val_img_paths, _, val_msk_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # Evaluate on 80 validation images for high robustness
    val_img_paths = val_img_paths[:80]
    val_msk_paths = val_msk_paths[:80]
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
        for idx in tqdm(range(len(val_img_paths)), desc="Synergistic"):
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
            
            # Optimal synergistic blend from sweep
            p_syn_blend = 0.4 * p_main_np + 0.3 * p_unet_np + 0.3 * p_dl_np
            p_syn_blend[0] = p_main_np[0] # keep class 0 unscaled
            
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(p_syn_blend[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            syn_probs_list.append(probs_resized)
            
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
        for idx in tqdm(range(len(val_img_paths)), desc="Meta-Stacked"):
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
            
    del meta_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- PHASE 3: JOINT GRID SEARCH ---
    print("\n--- Phase 3: Joint Grid Search over Weights, Threshold, and Min Area ---")
    
    w_syn_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    thresh_list = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    min_areas = [0, 8, 15, 25, 35, 50, 75]
    
    results = []
    
    # Loop over all configurations
    for w_syn in w_syn_list:
        w_meta = 1.0 - w_syn
        for thresh in thresh_list:
            for area in min_areas:
                row_dices = {c: [] for c in range(10)}
                for syn_p, meta_p, target in zip(syn_probs_list, meta_probs_list, targets_list):
                    probs = w_syn * syn_p + w_meta * meta_p
                    
                    pred_labels = np.argmax(probs, axis=0)
                    class0_mask = (pred_labels == 0)
                    fallback_labels = np.argmax(probs[1:], axis=0) + 1
                    
                    low_conf = class0_mask & (probs[0] < thresh)
                    if np.any(low_conf):
                        pred_labels[low_conf] = fallback_labels[low_conf]
                        
                    if area > 0:
                        pred_labels = neighbor_fill_cleanup(pred_labels.astype(np.uint8), min_area=area)
                        
                    valid_mask = (target != 255)
                    for c in range(10):
                        pred_c = (pred_labels == c) & valid_mask
                        tgt_c = (target == c) & valid_mask
                        row_dices[c].append(row_wise_dice(pred_c, tgt_c))
                        
                class_means = {c: np.mean(row_dices[c]) for c in range(10)}
                overall_mean = np.mean([class_means[c] for c in range(10)])
                
                results.append({
                    'w_syn': w_syn,
                    'w_meta': w_meta,
                    'thresh': thresh,
                    'area': area,
                    'mean_dice': overall_mean,
                    'class_0': class_means[0]
                })
                
    results = sorted(results, key=lambda x: x['mean_dice'], reverse=True)
    
    print("\n" + "="*80)
    print("TOP 25 ENSEMBLE CONFIGURATIONS (EVALUATED ON 80 VALIDATION IMAGES)")
    print("="*80)
    for idx, r in enumerate(results[:25]):
        print(f"{idx+1:02d}. Weight Syn: {r['w_syn']:.2f} | Weight Meta: {r['w_meta']:.2f} | Thresh C0: {r['thresh']:.2f} | Area: {r['area']:2d} || Mean Dice: {r['mean_dice']:.5f} | Class 0 Dice: {r['class_0']:.5f}")
    print("="*80)

if __name__ == '__main__':
    main()
