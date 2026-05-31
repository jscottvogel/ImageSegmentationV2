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
    indices = np.random.choice(len(tr_img), 150, replace=False)[:50] # Use 50 images for solid comparison
    
    dices_a = []
    dices_b = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating"):
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
            
            # --- Method A: Resize Probabilities (Bilinear) -> Argmax ---
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(p_blend[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            fallback_a = np.argmax(probs_resized[1:], axis=0) + 1
            pred_a = np.argmax(probs_resized, axis=0).astype(np.uint8)
            c0_mask_a = (pred_a == 0)
            low_conf_a = c0_mask_a & (probs_resized[0] < 0.99)
            if np.any(low_conf_a):
                pred_a[low_conf_a] = fallback_a[low_conf_a].astype(np.uint8)
            pred_a = neighbor_fill_cleanup(pred_a, min_area=100)
            
            # --- Method B: Argmax (Low Res) -> Resize Mask (Nearest) ---
            # Do background thresholding at low resolution first
            fallback_low = np.argmax(p_blend[1:], axis=0) + 1
            pred_low = np.argmax(p_blend, axis=0).astype(np.uint8)
            c0_mask_low = (pred_low == 0)
            low_conf_low = c0_mask_low & (p_blend[0] < 0.99)
            if np.any(low_conf_low):
                pred_low[low_conf_low] = fallback_low[low_conf_low].astype(np.uint8)
            
            # Resize the mask using nearest neighbor
            pred_b = cv2.resize(pred_low, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            pred_b = neighbor_fill_cleanup(pred_b, min_area=100)
            
            # Calculate Dice for both
            valid_mask = (gt_label != 255)
            
            # Method A score
            img_dices_a = []
            for c in range(10):
                pred_c = (pred_a == c) & valid_mask
                gt_c = (gt_label == c) & valid_mask
                intersection = np.sum(pred_c & gt_c)
                if np.sum(gt_c) == 0:
                    dice = 1.0 if np.sum(pred_c) == 0 else 0.0
                else:
                    dice = (2. * intersection) / (np.sum(pred_c) + np.sum(gt_c) + 1e-6)
                img_dices_a.append(dice)
            dices_a.append(np.mean(img_dices_a))
            
            # Method B score
            img_dices_b = []
            for c in range(10):
                pred_c = (pred_b == c) & valid_mask
                gt_c = (gt_label == c) & valid_mask
                intersection = np.sum(pred_c & gt_c)
                if np.sum(gt_c) == 0:
                    dice = 1.0 if np.sum(pred_c) == 0 else 0.0
                else:
                    dice = (2. * intersection) / (np.sum(pred_c) + np.sum(gt_c) + 1e-6)
                img_dices_b.append(dice)
            dices_b.append(np.mean(img_dices_b))
            
    print(f"\n--- Resizing Logic Comparison (50 Images) ---")
    print(f"Method A: Resize Probabilities (Bilinear): {np.mean(dices_a):.5f}")
    print(f"Method B: Argmax First (Nearest Mask):   {np.mean(dices_b):.5f}")

if __name__ == '__main__':
    main()
