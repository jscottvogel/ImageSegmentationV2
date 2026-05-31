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
from scratch.eval_advanced_tta import neighbor_fill_cleanup

def evaluate_sigma(sigma):
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
                
            # Gaussian blur on probability maps at native resolution
            if sigma > 0:
                for c in range(10):
                    probs_resized[c] = gaussian_filter(probs_resized[c], sigma=sigma)
                probs_resized /= np.sum(probs_resized, axis=0, keepdims=True) + 1e-8
                
            fallback = np.argmax(probs_resized[1:], axis=0) + 1
            pred_labels = np.argmax(probs_resized, axis=0).astype(np.uint8)
            c0_mask = (pred_labels == 0)
            low_conf_mask = c0_mask & (probs_resized[0] < 0.99)
            if np.any(low_conf_mask):
                pred_labels[low_conf_mask] = fallback[low_conf_mask].astype(np.uint8)
                
            pred_labels = neighbor_fill_cleanup(pred_labels, min_area=100)
            
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
    print("Evaluating Gaussian filter sigma values on validation set (No TTA + Blending)...")
    for s in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        score = evaluate_sigma(s)
        print(f"Sigma: {s:<5.1f} | Score: {score:.5f}")

if __name__ == '__main__':
    main()
