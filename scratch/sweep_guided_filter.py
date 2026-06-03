import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from synergistic_model import FloodNetSynergisticNet
from competitive_model import FloodNetCompetitiveModel
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask
from scratch.eval_advanced_tta import neighbor_fill_cleanup

def guided_filter(I, p, r, eps):
    # Convert guidance to grayscale
    I_gray = cv2.cvtColor((I * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    H, W = I_gray.shape
    C = p.shape[0]
    
    q = np.zeros_like(p)
    w_size = (2 * r + 1, 2 * r + 1)
    
    mean_I = cv2.blur(I_gray, w_size)
    mean_II = cv2.blur(I_gray * I_gray, w_size)
    var_I = mean_II - mean_I * mean_I
    
    for c in range(C):
        pc = p[c]
        mean_p = cv2.blur(pc, w_size)
        mean_Ip = cv2.blur(I_gray * pc, w_size)
        cov_Ip = mean_Ip - mean_I * mean_p
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = cv2.blur(a, w_size)
        mean_b = cv2.blur(b, w_size)
        
        q[c] = mean_a * I_gray + mean_b
        
    return q

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
    
    # Load Models
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
    
    np.random.seed(123)
    indices = np.random.choice(len(val_image_paths), 30, replace=False)
    val_sub_images = [val_image_paths[i] for i in indices]
    val_sub_masks = [val_mask_paths[i] for i in indices]
    
    gts = []
    imgs_rgb = []
    for idx in range(len(val_sub_images)):
        img = cv2.cvtColor(cv2.imread(val_sub_images[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.cvtColor(cv2.imread(val_sub_masks[idx]), cv2.COLOR_BGR2RGB)
        gt = rgb_to_mask(msk, id2color, 10)
        gts.append(gt)
        imgs_rgb.append(img)
        
    thresh_dict = {0: 0.95, 3: 0.50, 1: 0.50}
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    print("Computing baseline raw probabilities...")
    raw_probs = []
    with torch.no_grad():
        for img in tqdm(imgs_rgb, desc="Base Predictions"):
            img_norm = (img.astype(np.float32) / 255.0 - mean) / std
            img_tensor = torch.tensor(img_norm.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            out_syn = model_syn(img_tensor)
            p_syn_main = F.softmax(out_syn['main_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_unet = F.softmax(out_syn['unet_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_dl = F.softmax(out_syn['deeplab_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn = 0.4 * p_syn_main + 0.3 * p_syn_unet + 0.3 * p_syn_dl
            p_syn[0] = p_syn_main[0]
            
            meta_logits = model_meta(img_tensor)
            p_meta = F.softmax(meta_logits, dim=1).cpu().squeeze(0).numpy()
            
            p_blend = 0.50 * p_syn + 0.50 * p_meta
            p_blend[0] = p_syn[0]
            raw_probs.append(p_blend)
            
    # Baseline Dice (no Guided Filter)
    base_preds = []
    for p in raw_probs:
        pred_labels = apply_multiclass_thresholding(p, thresh_dict)
        pred_labels = neighbor_fill_cleanup(pred_labels.astype(np.uint8), min_area=128)
        base_preds.append(pred_labels)
    m_dice_base, dice_base = evaluate_predictions(gts, base_preds)
    print(f"\nBASELINE Macro Dice: {m_dice_base:.6f}")
    print(f"  Class 1: {dice_base[1]:.6f} | Class 3: {dice_base[3]:.6f}")
    
    # Sweep Grid
    r_vals = [1, 2, 4, 8]
    eps_vals = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    best_dice = m_dice_base
    best_r = None
    best_eps = None
    
    for r in r_vals:
        for eps in eps_vals:
            filtered_preds = []
            for i in range(len(imgs_rgb)):
                img = imgs_rgb[i]
                p = raw_probs[i]
                p_gf = guided_filter(img.astype(np.float32)/255.0, p, r=r, eps=eps)
                pred_labels = apply_multiclass_thresholding(p_gf, thresh_dict)
                pred_labels = neighbor_fill_cleanup(pred_labels.astype(np.uint8), min_area=128)
                filtered_preds.append(pred_labels)
                
            m_dice, dice_per_class = evaluate_predictions(gts, filtered_preds)
            print(f"r={r:2d}, eps={eps:.1e} -> Macro Dice: {m_dice:.6f} | C1: {dice_per_class[1]:.6f} | C3: {dice_per_class[3]:.6f}")
            
            if m_dice > best_dice:
                best_dice = m_dice
                best_r = r
                best_eps = eps
                
    print(f"\nBest configuration:")
    print(f"  r = {best_r}")
    print(f"  eps = {best_eps}")
    print(f"  Macro Dice: {best_dice:.6f} (Delta: {best_dice - m_dice_base:+.6f})")

if __name__ == '__main__':
    main()
