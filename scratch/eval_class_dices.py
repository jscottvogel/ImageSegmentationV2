import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import glob
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2

from synergistic_model import FloodNetSynergisticNet
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask
from scratch.eval_advanced_tta import neighbor_fill_cleanup

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FloodNetSynergisticNet(num_classes=10).to(device)
    weights_path = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=True)
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
        model.load_state_dict(state)
    model.eval()
    
    tr_img = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    tr_msk = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    np.random.seed(42)
    indices = np.random.choice(len(tr_img), 150, replace=False) # Use all 150 validation split images for robust stats!
    
    class_intersections = np.zeros(10, dtype=np.float64)
    class_unions = np.zeros(10, dtype=np.float64)
    class_gt_sums = np.zeros(10, dtype=np.float64)
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating classes"):
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
            
            out = model(img_tensor)
            
            p_main = F.softmax(out['main_output'], dim=1).cpu().squeeze(0).numpy()
            p_unet = F.softmax(out['unet_output'], dim=1).cpu().squeeze(0).numpy()
            p_deeplab = F.softmax(out['deeplab_output'], dim=1).cpu().squeeze(0).numpy()
            
            # Blend
            p_blend = 0.6 * p_main + 0.2 * p_unet + 0.2 * p_deeplab
            p_blend[0] = p_main[0]
            
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(p_blend[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                
            fallback = np.argmax(probs_resized[1:], axis=0) + 1
            pred_labels = np.argmax(probs_resized, axis=0).astype(np.uint8)
            c0_mask = (pred_labels == 0)
            low_conf_mask = c0_mask & (probs_resized[0] < 0.99)
            if np.any(low_conf_mask):
                pred_labels[low_conf_mask] = fallback[low_conf_mask].astype(np.uint8)
                
            pred_labels = neighbor_fill_cleanup(pred_labels, min_area=100)
            
            valid_mask = (gt_label != 255)
            for c in range(10):
                pred_c = (pred_labels == c) & valid_mask
                gt_c = (gt_label == c) & valid_mask
                
                class_intersections[c] += np.sum(pred_c & gt_c)
                class_unions[c] += np.sum(pred_c) + np.sum(gt_c)
                class_gt_sums[c] += np.sum(gt_c)
                
    print("\n--- Per-Class Global Dice Scores (Validation Set) ---")
    class_names = [
        "Background (0)", "Building-flooded (1)", "Building-non-flooded (2)", 
        "Road-flooded (3)", "Road-non-flooded (4)", "Water (5)", 
        "Tree (6)", "Vehicle (7)", "Pool (8)", "Grass (9)"
    ]
    
    mean_dice_1_9 = []
    for c in range(10):
        dice = (2. * class_intersections[c]) / (class_unions[c] + 1e-6)
        print(f"Class {c}: {class_names[c]:30s} | Dice: {dice:.5f} | GT Pixel Count: {int(class_gt_sums[c])}")
        if c >= 1:
            mean_dice_1_9.append(dice)
            
    print(f"\nMean Dice (Classes 1-9): {np.mean(mean_dice_1_9):.5f}")

if __name__ == '__main__':
    main()
