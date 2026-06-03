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
from scipy.ndimage import distance_transform_edt

from synergistic_model import FloodNetSynergisticNet
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask

def row_wise_dice(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    sum_pred = np.sum(pred_mask)
    sum_tgt = np.sum(target_mask)
    if sum_pred == 0 and sum_tgt == 0:
        return 1.0
    if sum_pred == 0 or sum_tgt == 0:
        return 0.0
    intersection = np.sum(pred_mask & target_mask)
    return (2.0 * intersection) / (sum_pred + sum_tgt)

def neighbor_fill_cleanup(pred_labels: np.ndarray, min_area=100) -> np.ndarray:
    if min_area <= 0:
        return pred_labels
    noise_mask = np.zeros(pred_labels.shape, dtype=bool)
    for class_id in range(1, 10):
        class_mask = (pred_labels == class_id).astype(np.uint8)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for i in range(1, num_components):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                noise_mask[labels == i] = True
    if not np.any(noise_mask):
        return pred_labels
    indices = distance_transform_edt(noise_mask, return_distances=False, return_indices=True)
    clean_labels = pred_labels[indices[0], indices[1]]
    return clean_labels

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Validation split
    image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    _, val_img_paths, _, val_msk_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # Evaluate on 100 validation images for robust statistics
    val_img_paths = val_img_paths[:100]
    val_msk_paths = val_msk_paths[:100]
    print(f"Evaluating on {len(val_img_paths)} validation images at 480x480 resolution.")
    
    model = FloodNetSynergisticNet(num_classes=10).to(device)
    weights_path = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        state = torch.load(weights_path, map_location=device, weights_only=True)
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
        model.load_state_dict(state)
    model.eval()
    
    # Precompute probs for TTA and Non-TTA
    probs_notta = []
    probs_tta = []
    targets = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_img_paths)), desc="Precomputing validation probabilities"):
            img_bgr = cv2.imread(val_img_paths[idx])
            
            msk = cv2.imread(val_msk_paths[idx])
            msk_rgb = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
            
            # Resize target mask to 480x480 for fast validation
            msk_resized = cv2.resize(msk_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            label = rgb_to_mask(msk_resized, id2color, 10)
            targets.append(label)
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            # Non-TTA forward pass
            out = model(img_tensor)
            p_main = F.softmax(out['main_output'], dim=1).squeeze(0).cpu().numpy()
            p_unet = F.softmax(out['unet_output'], dim=1).squeeze(0).cpu().numpy()
            p_dl = F.softmax(out['deeplab_output'], dim=1).squeeze(0).cpu().numpy()
            
            # TTA (horizontal flip) pass
            img_flipped = torch.flip(img_tensor, dims=[3])
            out_flip = model(img_flipped)
            
            p_main_flip = F.softmax(out_flip['main_output'], dim=1)
            p_main_unflip = torch.flip(p_main_flip, dims=[3]).squeeze(0).cpu().numpy()
            p_main_tta = 0.5 * (p_main + p_main_unflip)
            
            p_unet_flip = F.softmax(out_flip['unet_output'], dim=1)
            p_unet_unflip = torch.flip(p_unet_flip, dims=[3]).squeeze(0).cpu().numpy()
            p_unet_tta = 0.5 * (p_unet + p_unet_unflip)
            
            p_dl_flip = F.softmax(out_flip['deeplab_output'], dim=1)
            p_dl_unflip = torch.flip(p_dl_flip, dims=[3]).squeeze(0).cpu().numpy()
            p_dl_tta = 0.5 * (p_dl + p_dl_unflip)
            
            # Store probs
            p_dict_notta = {'main': p_main, 'unet': p_unet, 'deeplab': p_dl}
            p_dict_tta = {'main': p_main_tta, 'unet': p_unet_tta, 'deeplab': p_dl_tta}
            
            probs_notta.append(p_dict_notta)
            probs_tta.append(p_dict_tta)
            
    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Sweep space
    blends = [
        (1.0, 0.0, 0.0),
        (0.8, 0.1, 0.1),
        (0.6, 0.2, 0.2),
        (0.5, 0.25, 0.25),
        (0.4, 0.3, 0.3),
        (0.34, 0.33, 0.33),
        (0.0, 0.5, 0.5),
    ]
    
    thresholds = [0.0, 0.9, 0.95, 0.98, 0.99, 0.995]
    # min_areas at 480x480 (scaled down by ~39 from original sizes)
    # Area 3-5 at 480x480 corresponds to ~120-200 at 3000x2000
    min_areas = [0, 1, 2, 3, 4, 5, 8]
    modes = ['No-TTA', 'TTA']
    
    results = []
    
    for mode in modes:
        probs_source = probs_tta if mode == 'TTA' else probs_notta
        for w_main, w_unet, w_dl in blends:
            for thresh in thresholds:
                for area in min_areas:
                    # Evaluate on all 100 images
                    dice_vals = []
                    for idx in range(len(val_img_paths)):
                        p_dict = probs_source[idx]
                        target = targets[idx]
                        
                        # Blend at 480x480
                        p_blend = w_main * p_dict['main'] + w_unet * p_dict['unet'] + w_dl * p_dict['deeplab']
                        p_blend[0] = p_dict['main'][0] # Keep main class 0 unscaled
                        
                        # Apply class argmax and thresholding
                        if thresh == 0.0:
                            pred_labels = np.argmax(p_blend, axis=0)
                        else:
                            pred_labels = np.argmax(p_blend, axis=0)
                            c0_mask = (pred_labels == 0)
                            low_conf = c0_mask & (p_blend[0] < thresh)
                            if np.any(low_conf):
                                fallback = np.argmax(p_blend[1:], axis=0) + 1
                                pred_labels[low_conf] = fallback[low_conf]
                                
                        if area > 0:
                            pred_labels = neighbor_fill_cleanup(pred_labels.astype(np.uint8), min_area=area)
                            
                        valid_mask = (target != 255)
                        img_dices = []
                        for c in range(1, 10):
                            pred_c = (pred_labels == c) & valid_mask
                            tgt_c = (target == c) & valid_mask
                            img_dices.append(row_wise_dice(pred_c, tgt_c))
                        dice_vals.append(np.mean(img_dices))
                        
                    mean_val_dice = np.mean(dice_vals)
                    results.append({
                        'mode': mode,
                        'w_main': w_main,
                        'w_unet': w_unet,
                        'w_dl': w_dl,
                        'thresh': thresh,
                        'area': area,
                        'dice': mean_val_dice
                    })
                        
    # Sort results
    results = sorted(results, key=lambda x: x['dice'], reverse=True)
    
    print("\n" + "="*70)
    print("TOP 15 POST-PROCESSING CONFIGURATIONS (EVALUATED AT 480x480)")
    print("="*70)
    for idx, r in enumerate(results[:15]):
        # Map scaled area back to original area for printing
        orig_area_estimate = r['area'] * 39
        print(f"{idx+1:02d}. Mode: {r['mode']:6s} | Blend: ({r['w_main']:.2f}, {r['w_unet']:.2f}, {r['w_dl']:.2f}) | Thresh C0: {r['thresh']:.3f} | Area (orig equivalent): {orig_area_estimate:3d} || Mean Dice: {r['dice']:.5f}")
    print("="*70)

if __name__ == '__main__':
    main()
