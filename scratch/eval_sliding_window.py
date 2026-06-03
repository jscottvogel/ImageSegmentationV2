import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import gc

from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig, id2color, rgb_to_mask

def load_weights_custom(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if 'n_averaged' in state:
        del state['n_averaged']
    clean_state = {}
    for k, v in state.items():
        clean_state[k.replace('module.', '').replace('_orig_mod.', '')] = v
    model.load_state_dict(clean_state, strict=True)
    model.eval()
    return model

# 2D Cosine window for blending patch edges smoothly
def get_blend_window(crop_size):
    # 1D cosine window
    w = np.sin(np.pi * np.arange(crop_size) / (crop_size - 1))
    # 2D window
    w2d = np.outer(w, w)
    return w2d.astype(np.float32)

def evaluate_sliding_window():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load DeepLabV3+ model
    model = CustomDeepLabV3Plus(num_classes=10).to(device)
    model = load_weights_custom(model, "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt", device)
    
    # 2. Get validation images (using 10 images for speed)
    tr_img = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    tr_msk = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    np.random.seed(42)
    indices = np.random.choice(len(tr_img), 150, replace=False)[:10]
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Parameters for sliding window
    crop_size = 1024
    stride = 512
    blend_window = get_blend_window(crop_size)
    
    baseline_dices = []
    sliding_window_dices = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating"):
            img_bgr = cv2.imread(tr_img[idx])
            orig_h, orig_w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            msk_rgb = cv2.cvtColor(cv2.imread(tr_msk[idx]), cv2.COLOR_BGR2RGB)
            gt_label = rgb_to_mask(msk_rgb, id2color, 10)
            valid_mask = (gt_label != 255)
            
            # --- BASELINE: FULL IMAGE RESIZE INFERENCE ---
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_tensor = img_resized.astype(np.float32) / 255.0
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            out_base = model(img_tensor)
            p_base = F.softmax(out_base['main_output'], dim=1).cpu().squeeze(0).numpy()
            
            # Upsample probabilities back to native high-res
            probs_base_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_base_resized[c] = cv2.resize(p_base[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            pred_base = np.argmax(probs_base_resized, axis=0)
            
            # --- SLIDING WINDOW CROP INFERENCE ---
            # Create global accumulation arrays
            global_probs = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            global_weights = np.zeros((orig_h, orig_w), dtype=np.float32)
            
            # Generate patch grid indices
            y_starts = list(range(0, orig_h - crop_size + 1, stride))
            if y_starts[-1] + crop_size < orig_h:
                y_starts.append(orig_h - crop_size)
            x_starts = list(range(0, orig_w - crop_size + 1, stride))
            if x_starts[-1] + crop_size < orig_w:
                x_starts.append(orig_w - crop_size)
                
            for y_start in y_starts:
                for x_start in x_starts:
                    # Extract crop
                    crop_rgb = img_rgb[y_start:y_start+crop_size, x_start:x_start+crop_size]
                    
                    # Resize crop to model expected input size
                    crop_resized = cv2.resize(crop_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
                    
                    # Normalize and convert to tensor
                    crop_tensor = crop_resized.astype(np.float32) / 255.0
                    crop_tensor = (crop_tensor - mean) / std
                    crop_tensor = torch.tensor(crop_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
                    
                    # Predict crop
                    out_crop = model(crop_tensor)
                    p_crop = F.softmax(out_crop['main_output'], dim=1).cpu().squeeze(0).numpy()
                    
                    # Upsample crop probabilities back to native crop size (crop_size x crop_size)
                    p_crop_upsampled = np.zeros((10, crop_size, crop_size), dtype=np.float32)
                    for c in range(10):
                        p_crop_upsampled[c] = cv2.resize(p_crop[c], (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
                    
                    # Apply blend window
                    p_crop_upsampled = p_crop_upsampled * blend_window[None, ...]
                    
                    # Accumulate
                    global_probs[:, y_start:y_start+crop_size, x_start:x_start+crop_size] += p_crop_upsampled
                    global_weights[y_start:y_start+crop_size, x_start:x_start+crop_size] += blend_window
                    
            # Normalize global probabilities
            global_probs /= (global_weights[None, ...] + 1e-8)
            pred_sliding = np.argmax(global_probs, axis=0)
            
            # --- EVALUATE DICE SCORES ---
            # Baseline Dice
            base_dices = []
            for c in range(10):
                pred_c = (pred_base == c) & valid_mask
                gt_c = (gt_label == c) & valid_mask
                if np.sum(gt_c) == 0:
                    dice = 1.0 if np.sum(pred_c) == 0 else 0.0
                else:
                    dice = (2. * np.sum(pred_c & gt_c)) / (np.sum(pred_c) + np.sum(gt_c))
                base_dices.append(dice)
            baseline_dices.append(np.mean(base_dices))
            
            # Sliding Window Dice
            sliding_dices = []
            for c in range(10):
                pred_c = (pred_sliding == c) & valid_mask
                gt_c = (gt_label == c) & valid_mask
                if np.sum(gt_c) == 0:
                    dice = 1.0 if np.sum(pred_c) == 0 else 0.0
                else:
                    dice = (2. * np.sum(pred_c & gt_c)) / (np.sum(pred_c) + np.sum(gt_c))
                sliding_dices.append(dice)
            sliding_window_dices.append(np.mean(sliding_dices))
            
            # Clean VRAM
            del img_tensor, out_base
            gc.collect()
            torch.cuda.empty_cache()
            
    print(f"\n================ SLIDING WINDOW EVALUATION RESULTS ================")
    print(f"Baseline Mean Dice:       {np.mean(baseline_dices):.5f}")
    print(f"Sliding Window Mean Dice: {np.mean(sliding_window_dices):.5f}")
    print(f"Absolute Gain:            {np.mean(sliding_window_dices) - np.mean(baseline_dices):+.5f}")

if __name__ == '__main__':
    evaluate_sliding_window()
