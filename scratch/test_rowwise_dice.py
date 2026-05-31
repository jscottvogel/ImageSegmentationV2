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

from synergistic_model import FloodNetSynergisticNet
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask

def row_wise_dice(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    # Compute binary dice for a single class map
    # 1.0 if both empty
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
    model = FloodNetSynergisticNet(num_classes=10).to(device)
    
    weights_path = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    state = torch.load(weights_path, map_location=device, weights_only=True)
    state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
    model.load_state_dict(state)
    model.eval()
    
    image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    _, val_img_paths, _, val_msk_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # Check first 100 validation images
    val_img_paths = val_img_paths[:100]
    val_msk_paths = val_msk_paths[:100]
    
    print(f"Loading predictions for {len(val_img_paths)} validation images...")
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_img_paths))):
            img_bgr = cv2.imread(val_img_paths[idx])
            orig_h, orig_w = img_bgr.shape[:2]
            
            msk = cv2.imread(val_msk_paths[idx])
            msk_rgb = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
            label = rgb_to_mask(msk_rgb, id2color, 10)
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            preds_dict = model(img_tensor)
            logits = preds_dict['main_output']
            
            img_flipped = torch.flip(img_tensor, dims=[3])
            preds_flipped_dict = model(img_flipped)
            logits_flipped = preds_flipped_dict['main_output']
            
            probs_std = F.softmax(logits, dim=1)
            probs_flip = F.softmax(logits_flipped, dim=1)
            probs_unflip = torch.flip(probs_flip, dims=[3])
            fused_probs = (probs_std + probs_unflip) * 0.5
            fused_probs_np = fused_probs.squeeze(0).cpu().numpy()
            
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(fused_probs_np[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                
            all_probs.append(probs_resized)
            all_targets.append(label)
            
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    
    print("\nRow-wise (Kaggle-style) Mean Dice scores:")
    for thresh in thresholds:
        row_dices = {c: [] for c in range(10)}
        
        for probs, target in zip(all_probs, all_targets):
            if thresh == 0.0:
                pred_labels = np.argmax(probs, axis=0)
            elif thresh == 1.0:
                pred_labels = np.argmax(probs[1:], axis=0) + 1
            else:
                pred_labels = np.argmax(probs, axis=0)
                class0_mask = (pred_labels == 0)
                low_conf_mask = class0_mask & (probs[0] < thresh)
                if np.any(low_conf_mask):
                    fallback_labels = np.argmax(probs[1:], axis=0) + 1
                    pred_labels[low_conf_mask] = fallback_labels[low_conf_mask]
            
            # Mask out index 255 (ignore) from evaluation
            valid_mask = (target != 255)
            
            for c in range(10):
                pred_c = (pred_labels == c) & valid_mask
                tgt_c = (target == c) & valid_mask
                row_dices[c].append(row_wise_dice(pred_c, tgt_c))
                
        class_means = {c: np.mean(row_dices[c]) for c in range(10)}
        overall_mean = np.mean([class_means[c] for c in range(10)])
        class_1_9_mean = np.mean([class_means[c] for c in range(1, 10)])
        
        print(f"\nThreshold: {thresh:.2f}")
        print(f"  Class 0 Row-wise Dice: {class_means[0]:.4f}")
        print(f"  Class 1-9 Row-wise Avg: {class_1_9_mean:.4f}")
        print(f"  Mean Row-wise Dice (All 10): {overall_mean:.4f}")

if __name__ == '__main__':
    main()
