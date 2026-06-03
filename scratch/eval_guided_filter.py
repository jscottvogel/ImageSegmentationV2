import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from synergistic_model import FloodNetSynergisticNet
from competitive_model import FloodNetCompetitiveModel
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask
from scratch.eval_advanced_tta import neighbor_fill_cleanup

def guided_filter(I, p, r, eps):
    """
    Fast Guided Image Filter using OpenCV boxFilter.
    I: guide image (H, W, 3) or (H, W), float32 in [0, 1]
    p: filtering input (H, W), float32
    r: local window radius
    eps: regularization parameter
    """
    if len(I.shape) == 3:
        I_gray = cv2.cvtColor((I * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        I_gray = I
        
    I_gray = I_gray.astype(np.float32)
    p = p.astype(np.float32)
    wsize = (2*r+1, 2*r+1)
    
    mean_I = cv2.boxFilter(I_gray, -1, wsize)
    mean_p = cv2.boxFilter(p, -1, wsize)
    mean_Ip = cv2.boxFilter(I_gray * p, -1, wsize)
    
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(I_gray * I_gray, -1, wsize)
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, -1, wsize)
    mean_b = cv2.boxFilter(b, -1, wsize)
    
    q = mean_a * I_gray + mean_b
    return q

def evaluate_cached_predictions(cached_data, r, eps, use_guided=True, min_area=8):
    class_intersections = np.zeros(10, dtype=np.float64)
    class_unions = np.zeros(10, dtype=np.float64)
    
    for item in cached_data:
        p_blend = item['p_blend']
        gt_label = item['gt_label']
        img_resized = item['img_resized']
        
        probs = p_blend.copy()
        
        if use_guided:
            I_guide = img_resized.astype(np.float32) / 255.0
            for c in range(10):
                probs[c] = guided_filter(I_guide, probs[c], r, eps)
            probs = np.clip(probs, 0, 1)
            probs /= (np.sum(probs, axis=0, keepdims=True) + 1e-8)
            
        fallback = np.argmax(probs[1:], axis=0) + 1
        pred_labels = np.argmax(probs, axis=0).astype(np.uint8)
        
        c0_mask = (pred_labels == 0)
        low_conf_mask = c0_mask & (probs[0] < 0.95)
        if np.any(low_conf_mask):
            pred_labels[low_conf_mask] = fallback[low_conf_mask].astype(np.uint8)
            
        pred_labels = neighbor_fill_cleanup(pred_labels, min_area=min_area)
        
        valid_mask = (gt_label != 255)
        for c in range(10):
            pred_c = (pred_labels == c) & valid_mask
            gt_c = (gt_label == c) & valid_mask
            
            class_intersections[c] += np.sum(pred_c & gt_c)
            class_unions[c] += np.sum(pred_c) + np.sum(gt_c)
            
    mean_dice_1_9 = []
    for c in range(1, 10):
        dice = (2. * class_intersections[c]) / (class_unions[c] + 1e-6)
        mean_dice_1_9.append(dice)
        
    return np.mean(mean_dice_1_9)

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
    
    tr_img = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    tr_msk = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    np.random.seed(42)
    indices = np.random.choice(len(tr_img), 30, replace=False)
    
    print("\nCaching model predictions in RAM for 30 validation images...")
    cached_data = []
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Predicting"):
            img_bgr = cv2.imread(tr_img[idx])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            msk_rgb = cv2.cvtColor(cv2.imread(tr_msk[idx]), cv2.COLOR_BGR2RGB)
            msk_resized = cv2.resize(msk_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            gt_label = rgb_to_mask(msk_resized, id2color, 10)
            
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            # Predict
            out_syn = model_syn(img_tensor)
            p_syn_main = F.softmax(out_syn['main_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_unet = F.softmax(out_syn['unet_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_dl = F.softmax(out_syn['deeplab_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_notta = 0.4 * p_syn_main + 0.3 * p_syn_unet + 0.3 * p_syn_dl
            p_syn_notta[0] = p_syn_main[0]
            
            meta_logits = model_meta(img_tensor)
            p_meta = F.softmax(meta_logits, dim=1).cpu().squeeze(0).numpy()
            
            p_blend = 0.30 * p_syn_notta + 0.70 * p_meta
            p_blend[0] = p_syn_notta[0]
            
            cached_data.append({
                'p_blend': p_blend,
                'gt_label': gt_label,
                'img_resized': img_resized
            })
            
    print("\nRunning Baseline (without Guided Filter)...")
    base_dice = evaluate_cached_predictions(cached_data, r=0, eps=0, use_guided=False)
    print(f"Baseline Mean Dice: {base_dice:.6f}")
    
    print("\nStarting Parameter Sweep for Guided Filter Edge Refinement...")
    best_dice = base_dice
    best_params = None
    
    for r in [1, 2, 4, 8]:
        for eps in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            dice = evaluate_cached_predictions(cached_data, r=r, eps=eps, use_guided=True)
            print(f"Guided Filter (r={r}, eps={eps:.5f}) -> Mean Dice: {dice:.6f} (Delta: {dice - base_dice:+.6f})")
            if dice > best_dice:
                best_dice = dice
                best_params = (r, eps)
                
    if best_params is not None:
        print(f"\nOptimal Edge Refinement Parameters Found: r={best_params[0]}, eps={best_params[1]:.5f} with Dice: {best_dice:.6f} (Delta: {best_dice - base_dice:+.6f})")
    else:
        print("\nNo parameter set outperformed the baseline.")

if __name__ == '__main__':
    main()
