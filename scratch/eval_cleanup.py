import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import glob
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
from scipy.ndimage import distance_transform_edt

from synergistic_model import FloodNetSynergisticNet
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask

def safe_morphological_cleanup(pred_labels: np.ndarray, min_area=50) -> np.ndarray:
    if min_area <= 0:
        return pred_labels
    clean_labels = pred_labels.copy()
    for class_id in np.unique(pred_labels):
        if class_id == 0: continue
        class_mask = (pred_labels == class_id).astype(np.uint8)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for i in range(1, num_components):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                clean_labels[labels == i] = 0
    return clean_labels

def neighbor_fill_cleanup(pred_labels: np.ndarray, min_area=50) -> np.ndarray:
    if min_area <= 0:
        return pred_labels
    
    # 1. Find all small noise regions of classes 1-9
    noise_mask = np.zeros(pred_labels.shape, dtype=bool)
    for class_id in range(1, 10):
        class_mask = (pred_labels == class_id).astype(np.uint8)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for i in range(1, num_components):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                noise_mask[labels == i] = True
                
    if not np.any(noise_mask):
        return pred_labels
        
    # 2. Fill noise regions with nearest valid pixel labels
    indices = distance_transform_edt(noise_mask, return_distances=False, return_indices=True)
    clean_labels = pred_labels[indices[0], indices[1]]
    return clean_labels

def evaluate_cleanup_strategy(strategy_func, thresh=0.99, min_area=50):
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
    # Using 50 validation images for speed
    indices = np.random.choice(len(tr_img), 150, replace=False)[:50]
    
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
            
            # Simple TTA
            preds_dict = model(img_tensor)
            out_std = preds_dict['main_output']
            probs_std = F.softmax(out_std, dim=1).cpu()
            
            img_flipped = torch.flip(img_tensor, dims=[3])
            out_flipped = model(img_flipped)['main_output']
            probs_flipped = F.softmax(out_flipped, dim=1).cpu()
            probs_unflipped = torch.flip(probs_flipped, dims=[3])
            
            fused_probs = ((probs_std + probs_unflipped) * 0.5).squeeze(0).numpy()
            
            # Resize
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(fused_probs[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # Class 0 Threshold
            pred_labels = np.argmax(probs_resized, axis=0).astype(np.uint8)
            c0_mask = (pred_labels == 0)
            low_conf_mask = c0_mask & (probs_resized[0] < thresh)
            if np.any(low_conf_mask):
                fallback = np.argmax(probs_resized[1:], axis=0) + 1
                pred_labels[low_conf_mask] = fallback[low_conf_mask].astype(np.uint8)
                
            # Apply Morphological Cleanup Strategy
            pred_labels = strategy_func(pred_labels, min_area=min_area)
            
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
    print("Evaluating different post-processing cleanup strategies on validation split...")
    
    # 1. No cleanup
    print("Evaluating: No Cleanup...")
    d_none = evaluate_cleanup_strategy(lambda x, min_area: x, min_area=0)
    print(f"--> Score: {d_none:.5f}\n")
    
    # 2. Classic safe_morphological_cleanup (to Class 0)
    for area in [50, 100, 200]:
        print(f"Evaluating: safe_morphological_cleanup (to Class 0, area={area})...")
        d_classic = evaluate_cleanup_strategy(safe_morphological_cleanup, min_area=area)
        print(f"--> Score: {d_classic:.5f}\n")
        
    # 3. Neighbor fill cleanup
    for area in [50, 100, 200, 500, 1000]:
        print(f"Evaluating: neighbor_fill_cleanup (area={area})...")
        d_neighbor = evaluate_cleanup_strategy(neighbor_fill_cleanup, min_area=area)
        print(f"--> Score: {d_neighbor:.5f}\n")

if __name__ == '__main__':
    main()
