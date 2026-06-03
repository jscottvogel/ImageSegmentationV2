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

def evaluate_configuration(cached_data, w_syn, thresh, use_guided, r, eps, min_area):
    class_intersections = np.zeros(10, dtype=np.float64)
    class_unions = np.zeros(10, dtype=np.float64)
    
    for item in cached_data:
        p_syn = item['p_syn']
        p_meta = item['p_meta']
        gt_label = item['gt_label']
        img_resized = item['img_resized']
        
        # Blend
        p_blend = w_syn * p_syn + (1.0 - w_syn) * p_meta
        p_blend[0] = p_syn[0] # keep class 0 unscaled
        
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
        low_conf_mask = c0_mask & (probs[0] < thresh)
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
            
            # Predict Synergistic Net with HFlip TTA
            out_syn = model_syn(img_tensor)
            p_syn_main = F.softmax(out_syn['main_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_unet = F.softmax(out_syn['unet_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_dl = F.softmax(out_syn['deeplab_output'], dim=1).cpu().squeeze(0).numpy()
            
            # Flipped pass
            img_flipped = torch.flip(img_tensor, dims=[3])
            out_syn_flip = model_syn(img_flipped)
            p_syn_main_flip = F.softmax(out_syn_flip['main_output'], dim=1)
            p_syn_main_flip_np = torch.flip(p_syn_main_flip, dims=[3]).cpu().squeeze(0).numpy()
            p_syn_unet_flip = F.softmax(out_syn_flip['unet_output'], dim=1)
            p_syn_unet_flip_np = torch.flip(p_syn_unet_flip, dims=[3]).cpu().squeeze(0).numpy()
            p_syn_dl_flip = F.softmax(out_syn_flip['deeplab_output'], dim=1)
            p_syn_dl_flip_np = torch.flip(p_syn_dl_flip, dims=[3]).cpu().squeeze(0).numpy()
            
            # TTA main outputs (average)
            p_syn_main_tta = (p_syn_main + p_syn_main_flip_np) * 0.5
            p_syn_unet_tta = (p_syn_unet + p_syn_unet_flip_np) * 0.5
            p_syn_dl_tta = (p_syn_dl + p_syn_dl_flip_np) * 0.5
            
            p_syn_tta = 0.4 * p_syn_main_tta + 0.3 * p_syn_unet_tta + 0.3 * p_syn_dl_tta
            p_syn_tta[0] = p_syn_main_tta[0]
            
            # Predict Meta Net (which has internal TTA/processing)
            meta_logits = model_meta(img_tensor)
            p_meta = F.softmax(meta_logits, dim=1).cpu().squeeze(0).numpy()
            
            cached_data.append({
                'p_syn': p_syn_tta,
                'p_meta': p_meta,
                'gt_label': gt_label,
                'img_resized': img_resized
            })
            
    print("\nStarting Hyperparameter Grid Search...")
    # Sweep space
    w_syn_list = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    thresh_list = [0.90, 0.95, 0.99]
    guided_list = [False, True]
    min_area_list = [25, 50, 100]
    
    best_dice = 0.0
    best_params = {}
    
    # Baseline for reference (w_syn=0.30, thresh=0.95, guided=False, min_area=50)
    ref_dice = evaluate_configuration(cached_data, w_syn=0.30, thresh=0.95, use_guided=False, r=1, eps=0.001, min_area=50)
    print(f"Reference Baseline Dice (w_syn=0.30, thresh=0.95, guided=False, area=50): {ref_dice:.6f}")
    
    total_runs = len(w_syn_list) * len(thresh_list) * len(guided_list) * len(min_area_list)
    print(f"Total Sweep Runs: {total_runs}")
    
    run_idx = 0
    for w in w_syn_list:
        for t in thresh_list:
            for g in guided_list:
                for a in min_area_list:
                    dice = evaluate_configuration(cached_data, w_syn=w, thresh=t, use_guided=g, r=1, eps=0.001, min_area=a)
                    if dice > best_dice:
                        best_dice = dice
                        best_params = {
                            'w_syn': w, 'thresh': t, 'guided': g, 'min_area': a
                        }
                    run_idx += 1
                    if run_idx % 20 == 0:
                        print(f"Progress: {run_idx}/{total_runs} | Current Best Dice: {best_dice:.6f}")
                        
    print("\n================= SWEEP RESULTS =================")
    print(f"Optimal Hyperparameters:")
    print(f"  - Synergistic Weight: {best_params['w_syn']:.2f} (Meta Weight: {1.0 - best_params['w_syn']:.2f})")
    print(f"  - Class-0 Background suppression threshold: {best_params['thresh']:.2f}")
    print(f"  - Use Guided Filter: {best_params['guided']}")
    print(f"  - Minimum Area morphological filter: {best_params['min_area']}")
    print(f"Best validation Mean Dice (Classes 1-9): {best_dice:.6f} (Delta: {best_dice - ref_dice:+.6f})")

if __name__ == '__main__':
    main()
