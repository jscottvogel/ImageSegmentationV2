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

def evaluate_configuration(cached_data, w_syn, thresh_c0, thresh_c3, thresh_c1, min_area):
    class_intersections = np.zeros(10, dtype=np.float64)
    class_unions = np.zeros(10, dtype=np.float64)
    
    thresh_dict = {0: thresh_c0, 3: thresh_c3, 1: thresh_c1}
    
    for item in cached_data:
        p_syn = item['p_syn']
        p_meta = item['p_meta']
        gt_label = item['gt_label']
        
        # Blend
        p_blend = w_syn * p_syn + (1.0 - w_syn) * p_meta
        p_blend[0] = p_syn[0] # keep class 0 unscaled
        
        pred_labels = apply_multiclass_thresholding(p_blend.copy(), thresh_dict)
        pred_labels = neighbor_fill_cleanup(pred_labels.astype(np.uint8), min_area=min_area)
        
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
    
    image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    from sklearn.model_selection import train_test_split
    _, val_image_paths, _, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    np.random.seed(42)
    indices = np.random.choice(len(val_image_paths), 100, replace=False)
    val_sub_images = [val_image_paths[i] for i in indices]
    val_sub_masks = [val_mask_paths[i] for i in indices]
    
    print("\nCaching model predictions (WITHOUT TTA) in RAM for 100 unseen validation images...")
    cached_data = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_sub_images)), desc="Predicting"):
            img_bgr = cv2.imread(val_sub_images[idx])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            msk_rgb = cv2.cvtColor(cv2.imread(val_sub_masks[idx]), cv2.COLOR_BGR2RGB)
            msk_resized = cv2.resize(msk_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            gt_label = rgb_to_mask(msk_resized, id2color, 10)
            
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            # Predict Synergistic Net
            out_syn = model_syn(img_tensor)
            p_syn_main = F.softmax(out_syn['main_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_unet = F.softmax(out_syn['unet_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_dl = F.softmax(out_syn['deeplab_output'], dim=1).cpu().squeeze(0).numpy()
            
            p_syn_notta = 0.4 * p_syn_main + 0.3 * p_syn_unet + 0.3 * p_syn_dl
            p_syn_notta[0] = p_syn_main[0]
            
            # Predict Meta Net
            meta_logits = model_meta(img_tensor)
            p_meta = F.softmax(meta_logits, dim=1).cpu().squeeze(0).numpy()
            
            cached_data.append({
                'p_syn': p_syn_notta,
                'p_meta': p_meta,
                'gt_label': gt_label
            })
            
    print("\nStarting Hyperparameter Grid Search (No-TTA)...")
    # Sweep space around our current best weights
    w_syn_list = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    thresh_c0_list = [0.90, 0.95, 1.00]
    thresh_c3_list = [0.30, 0.40, 0.50, 0.60]
    thresh_c1_list = [0.30, 0.40, 0.50, 0.60]
    min_area_list = [128]
    
    best_dice = 0.0
    best_params = {}
    
    # Reference Baseline (w_syn=0.50, thresh_c0=0.95, thresh_c3=0.50, thresh_c1=0.50, min_area=128)
    ref_dice = evaluate_configuration(cached_data, w_syn=0.50, thresh_c0=0.95, thresh_c3=0.50, thresh_c1=0.50, min_area=128)
    print(f"Reference Baseline Dice (w_syn=0.50, c0=0.95, c3=0.50, c1=0.50, area=128): {ref_dice:.6f}")
    
    total_runs = len(w_syn_list) * len(thresh_c0_list) * len(thresh_c3_list) * len(thresh_c1_list) * len(min_area_list)
    print(f"Total Sweep Runs: {total_runs}")
    
    run_idx = 0
    for w in w_syn_list:
        for c0 in thresh_c0_list:
            for c3 in thresh_c3_list:
                for c1 in thresh_c1_list:
                    for a in min_area_list:
                        dice = evaluate_configuration(cached_data, w_syn=w, thresh_c0=c0, thresh_c3=c3, thresh_c1=c1, min_area=a)
                        if dice > best_dice:
                            best_dice = dice
                            best_params = {
                                'w_syn': w, 'thresh_c0': c0, 'thresh_c3': c3, 'thresh_c1': c1, 'min_area': a
                            }
                        run_idx += 1
                        if run_idx % 50 == 0:
                            print(f"Progress: {run_idx}/{total_runs} | Current Best Dice: {best_dice:.6f}")
                            
    print("\n================= SWEEP RESULTS =================")
    print(f"Optimal Hyperparameters:")
    print(f"  - Synergistic Weight: {best_params['w_syn']:.2f} (Meta Weight: {1.0 - best_params['w_syn']:.2f})")
    print(f"  - Class-0 Background threshold: {best_params['thresh_c0']:.2f}")
    print(f"  - Class-3 Flooded Road threshold: {best_params['thresh_c3']:.2f}")
    print(f"  - Class-1 Flooded Building threshold: {best_params['thresh_c1']:.2f}")
    print(f"  - Minimum Area: {best_params['min_area']}")
    print(f"Best validation Mean Dice (Classes 1-9): {best_dice:.6f} (Delta: {best_dice - ref_dice:+.6f})")

if __name__ == '__main__':
    main()
