import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.ndimage import distance_transform_edt
from multiprocessing import Pool
import gc

from synergistic_model import FloodNetSynergisticNet
from competitive_model import FloodNetCompetitiveModel
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask

def row_wise_dice(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    sum_pred = np.sum(pred_mask)
    sum_tgt = np.sum(target_mask)
    if sum_pred == 0 and sum_tgt == 0:
        return 1.0
    if sum_pred == 0 or sum_tgt == 0:
        return 0.0
    intersection = np.sum(pred_mask & target_mask)
    return (2.0 * intersection) / (sum_pred + sum_tgt + 1e-8)

def neighbor_fill_cleanup_class_specific(pred_labels: np.ndarray, class_areas: list) -> np.ndarray:
    noise_mask = np.zeros(pred_labels.shape, dtype=bool)
    for class_id in range(10):
        area = class_areas[class_id]
        if area <= 0:
            continue
        class_mask = (pred_labels == class_id).astype(np.uint8)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for i in range(1, num_components):
            if stats[i, cv2.CC_STAT_AREA] < area:
                noise_mask[labels == i] = True
                
    if not np.any(noise_mask):
        return pred_labels
        
    indices = distance_transform_edt(noise_mask, return_distances=False, return_indices=True)
    clean_labels = pred_labels[indices[0], indices[1]]
    return clean_labels

def worker_cleanup_and_eval(args):
    idx, pred_raw, class_areas, target = args
    pred_clean = neighbor_fill_cleanup_class_specific(pred_raw, class_areas)
    
    valid_mask = (target != 255)
    img_dices = []
    for c in range(10):
        pred_c = (pred_clean == c) & valid_mask
        tgt_c = (target == c) & valid_mask
        img_dices.append(row_wise_dice(pred_c, tgt_c))
        
    return np.mean(img_dices)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load optimal config
    config_path = "model_checkpoint/ensemble_kaggle_config.pt"
    if not os.path.exists(config_path):
        print(f"Error: config not found at {config_path}!")
        return
        
    config = torch.load(config_path)
    best_w = config['best_w']
    best_t = config['best_t']
    
    print("\nLoaded Optimal Hyperparameters:")
    print("  - Weights (w_syn):", [round(w, 4) for w in best_w])
    print("  - Thresholds:", [round(t, 4) for t in best_t])
    
    # Validation split
    image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    _, val_image_paths, _, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # 400 validation images
    val_image_paths = val_image_paths[:400]
    val_mask_paths = val_mask_paths[:400]
    num_val = len(val_image_paths)
    print(f"Total Validation Images for Sweep: {num_val}")
    
    print("Precomputing raw predictions...")
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
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    raw_predictions = []
    targets_list = []
    
    best_t_arr = np.array(best_t)
    fallback_classes = [c for c in range(10) if best_t_arr[c] == 0.0]
    if len(fallback_classes) == 0:
        fallback_classes = [2]
        
    with torch.no_grad():
        for idx in tqdm(range(num_val), desc="Inference"):
            img_bgr = cv2.imread(val_image_paths[idx])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            msk_rgb = cv2.cvtColor(cv2.imread(val_mask_paths[idx]), cv2.COLOR_BGR2RGB)
            
            # Ground truth
            msk_resized = cv2.resize(msk_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            gt_label = rgb_to_mask(msk_resized, id2color, 10)
            targets_list.append(gt_label)
            
            # Input preprocessing
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
            x = torch.tensor(img_norm.transpose(2, 0, 1)[None, ...], dtype=torch.float32).to(device)
            
            # Predict Synergistic Net
            out_syn = model_syn(x)
            p_syn_main = F.softmax(out_syn['main_output'], dim=1)
            p_syn_unet = F.softmax(out_syn['unet_output'], dim=1)
            p_syn_dl = F.softmax(out_syn['deeplab_output'], dim=1)
            p_syn = 0.4 * p_syn_main + 0.3 * p_syn_unet + 0.3 * p_syn_dl
            p_syn[:, 0] = p_syn_main[:, 0]
            p_syn_np = p_syn.cpu().squeeze(0).numpy()
            
            # Predict Meta Net
            meta_logits = model_meta(x)
            p_meta_np = F.softmax(meta_logits, dim=1).cpu().squeeze(0).numpy()
            
            # Blend
            best_w_arr = np.array(best_w).reshape(10, 1, 1)
            probs = best_w_arr * p_syn_np + (1.0 - best_w_arr) * p_meta_np
            probs[0] = p_syn_np[0]
            
            # Multiclass thresholding fallback logic
            pred_labels = np.argmax(probs, axis=0)
            
            fallback_probs = probs[fallback_classes]
            fallback_idx = np.argmax(fallback_probs, axis=0)
            fallback = np.array(fallback_classes)[fallback_idx]
            
            for c in range(10):
                t = best_t_arr[c]
                if t > 0.0:
                    mask = (pred_labels == c) & (probs[c] < t)
                    pred_labels[mask] = fallback[mask]
                    
            raw_predictions.append(pred_labels)
            del x, out_syn, p_syn_main, p_syn_unet, p_syn_dl, p_syn, meta_logits
            
    del model_syn, model_meta
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nPrecomputation complete!")
    
    # 2. Greedy optimization of class-specific areas
    pool = Pool(processes=8)
    
    # Start from baseline configuration or global 96
    current_areas = config.get('best_areas', [96] * 10)
    print(f"\nStarting areas: {current_areas}")
    
    # Eval initial configuration
    eval_inputs = [(i, raw_predictions[i], current_areas, targets_list[i]) for i in range(num_val)]
    results = pool.map(worker_cleanup_and_eval, eval_inputs)
    best_score = np.mean(results)
    print(f"Initial macro Dice on 400 images: {best_score:.6f}")
    
    # Search space
    candidate_areas = [0, 8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 384, 512]
    
    improved = True
    iterations = 0
    max_iterations = 3
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        print(f"\n--- Coordinate Descent Pass {iterations} ---")
        
        # Loop classes in order of increasing physical scale to prioritize rare classes
        class_order = [7, 8, 3, 1, 2, 4, 5, 6, 9, 0]
        
        for c in class_order:
            best_area_for_c = current_areas[c]
            best_score_for_c = best_score
            
            print(f"Sweeping Class {c}... ", end="", flush=True)
            
            for area in candidate_areas:
                temp_areas = list(current_areas)
                temp_areas[c] = area
                
                eval_inputs = [(i, raw_predictions[i], temp_areas, targets_list[i]) for i in range(num_val)]
                results = pool.map(worker_cleanup_and_eval, eval_inputs)
                score = np.mean(results)
                
                if score > best_score_for_c + 1e-7:
                    best_score_for_c = score
                    best_area_for_c = area
                    
            if best_area_for_c != current_areas[c]:
                print(f"Improved Area: {current_areas[c]} -> {best_area_for_c} | Score: {best_score_for_c:.6f}")
                current_areas[c] = best_area_for_c
                best_score = best_score_for_c
                improved = True
            else:
                print(f"Unchanged ({current_areas[c]}) | Score: {best_score:.6f}")
                
    pool.close()
    pool.join()
    
    print("\n================ OPTIMIZATION RESULTS ================")
    print(f"Optimal Class-specific Morphological Areas on 400 images:")
    for c in range(10):
        print(f"  Class {c}: {current_areas[c]}")
    print(f"Final Validation Dice: {best_score:.6f}")
    
    # Save optimized config
    config['best_areas'] = current_areas
    torch.save(config, config_path)
    print(f"\nConfiguration saved successfully to {config_path}")

if __name__ == '__main__':
    main()
