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

def evaluate_multiscale(scales, scale_weights, thresh=0.99):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the synergistic model
    model = FloodNetSynergisticNet(num_classes=10).to(device)
    weights_path = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=True)
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
        model.load_state_dict(state)
    model.eval()
    
    # 2. Get validation data paths
    tr_img = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    tr_msk = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    np.random.seed(42)
    # Using first 50 validation images for faster benchmarking
    indices = np.random.choice(len(tr_img), 150, replace=False)[:50]
    
    row_dices = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc=f"Evaluating scales {scales}", leave=False):
            img_bgr = cv2.imread(tr_img[idx])
            orig_h, orig_w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            
            # Load ground truth mask
            msk_rgb = cv2.cvtColor(cv2.imread(tr_msk[idx]), cv2.COLOR_BGR2RGB)
            msk_resized = cv2.resize(msk_rgb, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            gt_label = rgb_to_mask(msk_resized, id2color, 10)
            
            # Normalize input
            img_tensor = resized_img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            fused_probs_accum = torch.zeros((1, 10, DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH), device='cpu')
            
            for scale, weight in zip(scales, scale_weights):
                scaled_h = int(DatasetConfig.IMG_HEIGHT * scale)
                scaled_w = int(DatasetConfig.IMG_WIDTH * scale)
                scaled_img = F.interpolate(img_tensor, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)
                
                # Standard pass
                out_std = model(scaled_img)['main_output']
                probs_std = F.softmax(out_std, dim=1).cpu()
                fused_probs_accum += F.interpolate(probs_std, size=(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH), mode='bilinear', align_corners=False) * weight * 0.5
                
                # Flip TTA pass
                img_flipped = torch.flip(scaled_img, dims=[3])
                out_flipped = model(img_flipped)['main_output']
                probs_flipped = F.softmax(out_flipped, dim=1).cpu()
                probs_unflipped = torch.flip(probs_flipped, dims=[3])
                fused_probs_accum += F.interpolate(probs_unflipped, size=(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH), mode='bilinear', align_corners=False) * weight * 0.5
                
            fused_probs = fused_probs_accum.squeeze(0).numpy()
            
            # Resize probabilities to native resolution
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(fused_probs[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # Threshold Class 0
            pred_labels = np.argmax(probs_resized, axis=0).astype(np.uint8)
            c0_mask = (pred_labels == 0)
            low_conf_mask = c0_mask & (probs_resized[0] < thresh)
            if np.any(low_conf_mask):
                fallback = np.argmax(probs_resized[1:], axis=0) + 1
                pred_labels[low_conf_mask] = fallback[low_conf_mask].astype(np.uint8)
            
            # Compute row-wise Dice score
            valid_mask = (gt_label != 255)
            img_dices = []
            for c in range(10):
                pred_c = (pred_labels == c) & valid_mask
                gt_c = (gt_label == c) & valid_mask
                
                intersection = np.sum(pred_c & gt_c)
                union = np.sum(pred_c | gt_c)
                
                if np.sum(gt_c) == 0:
                    dice = 1.0 if np.sum(pred_c) == 0 else 0.0
                else:
                    dice = (2. * intersection) / (np.sum(pred_c) + np.sum(gt_c) + 1e-6)
                img_dices.append(dice)
            
            row_dices.append(np.mean(img_dices))
            
    return np.mean(row_dices)

def main():
    print("Benchmarking different scales and weights for Multiscale TTA on validation split...")
    
    # 1. Base TTA (Scale 1.0)
    d1 = evaluate_multiscale([1.0], [1.0], thresh=0.99)
    print(f"Scale 1.0 TTA: {d1:.5f}")
    
    # 2. Multi-Scale TTA (0.875, 1.0, 1.125)
    d2 = evaluate_multiscale([0.875, 1.0, 1.125], [0.2, 0.6, 0.2], thresh=0.99)
    print(f"Scales [0.875, 1.0, 1.125] TTA: {d2:.5f}")
    
    # 3. Multi-Scale TTA (0.75, 1.0, 1.25)
    d3 = evaluate_multiscale([0.75, 1.0, 1.25], [0.15, 0.70, 0.15], thresh=0.99)
    print(f"Scales [0.75, 1.0, 1.25] TTA: {d3:.5f}")

if __name__ == '__main__':
    main()
