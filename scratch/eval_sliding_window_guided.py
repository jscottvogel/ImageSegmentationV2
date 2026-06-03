import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
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

def fast_guided_filter(I, p, r, eps, s=4):
    """
    Fast Guided Filter (He & Sun 2015) using subsampling factor s.
    I: guidance image, shape (H, W, 3) in [0, 1]
    p: filtering input (probability map), shape (C, H, W) in [0, 1]
    r: window radius in original scale
    eps: regularization parameter
    s: downsampling factor (e.g., 2 or 4)
    """
    if len(I.shape) == 3:
        I_gray = cv2.cvtColor((I * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        I_gray = I.astype(np.float32)
        
    H, W = I_gray.shape
    C = p.shape[0]
    
    # Subsampled dimensions
    H_sub = int(round(H / s))
    W_sub = int(round(W / s))
    
    I_sub = cv2.resize(I_gray, (W_sub, H_sub), interpolation=cv2.INTER_LINEAR)
    
    # Scale radius
    r_sub = max(1, int(round(r / s)))
    w_size_sub = (2 * r_sub + 1, 2 * r_sub + 1)
    
    # Guidance stats in subsampled space
    mean_I_sub = cv2.blur(I_sub, w_size_sub)
    mean_II_sub = cv2.blur(I_sub * I_sub, w_size_sub)
    var_I_sub = mean_II_sub - mean_I_sub * mean_I_sub
    
    q = np.zeros_like(p)
    
    for c in range(C):
        pc = p[c]
        pc_sub = cv2.resize(pc, (W_sub, H_sub), interpolation=cv2.INTER_LINEAR)
        
        mean_p_sub = cv2.blur(pc_sub, w_size_sub)
        mean_Ip_sub = cv2.blur(I_sub * pc_sub, w_size_sub)
        cov_Ip_sub = mean_Ip_sub - mean_I_sub * mean_p_sub
        
        # Coefficients a and b in subsampled space
        a_sub = cov_Ip_sub / (var_I_sub + eps)
        b_sub = mean_p_sub - a_sub * mean_I_sub
        
        # Mean of coefficients in subsampled space
        mean_a_sub = cv2.blur(a_sub, w_size_sub)
        mean_b_sub = cv2.blur(b_sub, w_size_sub)
        
        # Bilinear upsample back to original resolution
        mean_a = cv2.resize(mean_a_sub, (W, H), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b_sub, (W, H), interpolation=cv2.INTER_LINEAR)
        
        q[c] = mean_a * I_gray + mean_b
        
    return q

def sliding_window_inference(model_syn, model_meta, img_rgb, device, patch_h=1500, patch_w=2000, overlap_h=0, overlap_w=0):
    H, W, _ = img_rgb.shape
    num_classes = 10
    
    # Calculate grid coords
    y_starts = []
    y = 0
    while y + patch_h <= H:
        y_starts.append(y)
        if y + patch_h == H:
            break
        y = min(y + patch_h - overlap_h, H - patch_h)
        
    x_starts = []
    x = 0
    while x + patch_w <= W:
        x_starts.append(x)
        if x + patch_w == W:
            break
        x = min(x + patch_w - overlap_w, W - patch_w)
        
    prob_accumulator = np.zeros((num_classes, H, W), dtype=np.float32)
    weight_accumulator = np.zeros((H, W), dtype=np.float32)
    
    patch_weight = np.ones((patch_h, patch_w), dtype=np.float32)
    if overlap_h > 0:
        ramp_y = np.linspace(0, 1, overlap_h, dtype=np.float32)
        patch_weight[:overlap_h, :] *= ramp_y[:, None]
        patch_weight[-overlap_h:, :] *= ramp_y[::-1][:, None]
    if overlap_w > 0:
        ramp_x = np.linspace(0, 1, overlap_w, dtype=np.float32)
        patch_weight[:, :overlap_w] *= ramp_x[None, :]
        patch_weight[:, -overlap_w:] *= ramp_x[::-1][None, :]
        
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    for y0 in y_starts:
        for x0 in x_starts:
            patch = img_rgb[y0:y0+patch_h, x0:x0+patch_w]
            patch_norm = (patch.astype(np.float32) / 255.0 - mean) / std
            patch_tensor = torch.tensor(patch_norm.transpose(2, 0, 1)[None, ...], dtype=torch.float32).to(device)
            
            # Predict Synergistic
            out_syn = model_syn(patch_tensor)
            p_syn_main = F.softmax(out_syn['main_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_unet = F.softmax(out_syn['unet_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_dl = F.softmax(out_syn['deeplab_output'], dim=1).cpu().squeeze(0).numpy()
            
            p_syn = 0.4 * p_syn_main + 0.3 * p_syn_unet + 0.3 * p_syn_dl
            p_syn[0] = p_syn_main[0]
            
            # Predict Meta
            meta_logits = model_meta(patch_tensor)
            p_meta = F.softmax(meta_logits, dim=1).cpu().squeeze(0).numpy()
            
            p_blend = 0.50 * p_syn + 0.50 * p_meta
            p_blend[0] = p_syn[0]
            
            prob_accumulator[:, y0:y0+patch_h, x0:x0+patch_w] += p_blend * patch_weight[None, ...]
            weight_accumulator[y0:y0+patch_h, x0:x0+patch_w] += patch_weight
            
    prob_accumulator /= (weight_accumulator[None, ...] + 1e-6)
    return prob_accumulator

def apply_multiclass_thresholding(probs: np.ndarray, thresh_dict: dict) -> np.ndarray:
    pred = np.argmax(probs, axis=0)
    non_suppressed_classes = [c for c in range(10) if c not in thresh_dict or thresh_dict[c] == 0]
    fallback_idx = np.argmax(probs[non_suppressed_classes], axis=0)
    fallback = np.array(non_suppressed_classes)[fallback_idx]
    
    for c, t in thresh_dict.items():
        if t > 0:
            mask = (pred == c) & (probs[c] < t)
            pred[mask] = fallback[mask]
    return pred

def evaluate_predictions(gt_labels, pred_labels_list):
    # gt_labels: list of (H, W)
    # pred_labels_list: list of (H, W)
    class_intersections = np.zeros(10, dtype=np.float64)
    class_unions = np.zeros(10, dtype=np.float64)
    
    for gt, pred in zip(gt_labels, pred_labels_list):
        valid_mask = (gt != 255)
        for c in range(10):
            pred_c = (pred == c) & valid_mask
            gt_c = (gt == c) & valid_mask
            class_intersections[c] += np.sum(pred_c & gt_c)
            class_unions[c] += np.sum(pred_c) + np.sum(gt_c)
            
    mean_dice_1_9 = []
    dice_per_class = {}
    for c in range(1, 10):
        dice = (2. * class_intersections[c]) / (class_unions[c] + 1e-6)
        mean_dice_1_9.append(dice)
        dice_per_class[c] = dice
        
    return np.mean(mean_dice_1_9), dice_per_class

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading models...")
    
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
    
    image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    _, val_image_paths, _, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # Choose a representative subset of 20 images from unseen validation set
    np.random.seed(123)
    indices = np.random.choice(len(val_image_paths), 20, replace=False)
    val_sub_images = [val_image_paths[i] for i in indices]
    val_sub_masks = [val_mask_paths[i] for i in indices]
    
    print(f"Loading {len(val_sub_images)} high-resolution ground truth masks...")
    gts = []
    imgs_rgb = []
    for idx in tqdm(range(len(val_sub_images)), desc="Loading GT"):
        img = cv2.cvtColor(cv2.imread(val_sub_images[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.cvtColor(cv2.imread(val_sub_masks[idx]), cv2.COLOR_BGR2RGB)
        gt = rgb_to_mask(msk, id2color, 10)
        gts.append(gt)
        imgs_rgb.append(img)
        
    thresh_dict = {0: 0.95, 3: 0.50, 1: 0.50}
    
    # ------------------ PRECOMPUTE: Raw Blended Probabilities ------------------
    print("\nPrecomputing raw blended probabilities...")
    raw_probs_hr = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    with torch.no_grad():
        for img in tqdm(imgs_rgb, desc="Config 1 (Baseline) & Precompute"):
            orig_h, orig_w = img.shape[:2]
            img_resized = cv2.resize(img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
            img_tensor = torch.tensor(img_norm.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            # Predict Synergistic
            out_syn = model_syn(img_tensor)
            p_syn_main = F.softmax(out_syn['main_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_unet = F.softmax(out_syn['unet_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_dl = F.softmax(out_syn['deeplab_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn = 0.4 * p_syn_main + 0.3 * p_syn_unet + 0.3 * p_syn_dl
            p_syn[0] = p_syn_main[0]
            
            # Predict Meta
            meta_logits = model_meta(img_tensor)
            p_meta = F.softmax(meta_logits, dim=1).cpu().squeeze(0).numpy()
            
            p_blend = 0.50 * p_syn + 0.50 * p_meta
            p_blend[0] = p_syn[0]
            
            # Resize probability maps to original shape
            p_blend_hr = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                p_blend_hr[c] = cv2.resize(p_blend[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            raw_probs_hr.append(p_blend_hr)
            
    # Calculate baseline
    preds_lr = []
    for p_blend_hr in raw_probs_hr:
        pred_labels = apply_multiclass_thresholding(p_blend_hr, thresh_dict)
        pred_labels = neighbor_fill_cleanup(pred_labels.astype(np.uint8), min_area=128)
        preds_lr.append(pred_labels)
            
    m_dice_lr, dice_lr = evaluate_predictions(gts, preds_lr)
    print(f"\nCONFIG 1 (Baseline) Macro Dice 1-9: {m_dice_lr:.6f}")
    print(f"  Class 1 (Buildings): {dice_lr[1]:.6f} | Class 3 (Roads): {dice_lr[3]:.6f}")
    
    # ------------------ SWEEP: Fast Guided Filter Parameters ------------------
    r_vals = [2, 4, 8, 16]
    eps_vals = [1e-6, 1e-4, 1e-2]
    s_vals = [2, 4, 8]
    
    print("\nRunning Parameter Sweep for Fast Guided Filter...")
    print(f"{'Radius (r)':<12} | {'Epsilon (eps)':<14} | {'Scale (s)':<10} | {'Macro Dice':<12} | {'Delta':<10} | {'C1 Dice':<10} | {'C3 Dice':<10}")
    print("-" * 96)
    
    best_dice = m_dice_lr
    best_config = None
    
    for r in r_vals:
        for eps in eps_vals:
            for s in s_vals:
                preds_gf = []
                for i, img in enumerate(imgs_rgb):
                    p_blend_hr = raw_probs_hr[i]
                    p_blend_hr_gf = fast_guided_filter(img.astype(np.float32)/255.0, p_blend_hr, r=r, eps=eps, s=s)
                    pred_labels = apply_multiclass_thresholding(p_blend_hr_gf, thresh_dict)
                    pred_labels = neighbor_fill_cleanup(pred_labels.astype(np.uint8), min_area=128)
                    preds_gf.append(pred_labels)
                
                m_dice, dice_gf = evaluate_predictions(gts, preds_gf)
                delta = m_dice - m_dice_lr
                
                print(f"{r:<12} | {eps:<14.1e} | {s:<10} | {m_dice:<12.6f} | {delta:<+10.6f} | {dice_gf[1]:<10.6f} | {dice_gf[3]:<10.6f}")
                
                if m_dice > best_dice:
                    best_dice = m_dice
                    best_config = (r, eps, s, dice_gf[1], dice_gf[3])
                    
    print("\n================== BENCHMARK SUMMARY ==================")
    print(f"Configuration 1 (Low-Res Baseline): Macro Dice: {m_dice_lr:.6f}")
    if best_config:
        r, eps, s, c1, c3 = best_config
        print(f"Best Fast Guided Filter Config (r={r}, eps={eps:.1e}, s={s}): Macro Dice: {best_dice:.6f} (Delta: {best_dice - m_dice_lr:+.6f})")
        print(f"  Class 1: {c1:.6f} | Class 3: {c3:.6f}")
    else:
        print("No Fast Guided Filter configuration outperformed the raw upsampling baseline.")

if __name__ == '__main__':
    main()
