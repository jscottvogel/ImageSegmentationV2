import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import gc
import csv

from synergistic_model import FloodNetSynergisticNet
from competitive_model import FloodNetCompetitiveModel
from optimized_pytorch_version import DatasetConfig
from scratch.eval_advanced_tta import neighbor_fill_cleanup

def mask2rle(img: np.ndarray) -> str:
    pixels = img.T.ravel()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    runs[::2] += 1
    if len(runs) == 0:
        return ""
    return ' '.join(map(str, runs))

def apply_multiclass_thresholding(probs: np.ndarray, thresh_arr: np.ndarray) -> np.ndarray:
    pred = np.argmax(probs, axis=0)
    fallback_classes = [c for c in range(10) if thresh_arr[c] == 0.0]
    if len(fallback_classes) == 0:
        fallback_classes = [2]
        
    fallback_idx = np.argmax(probs[fallback_classes], axis=0)
    fallback = np.array(fallback_classes)[fallback_idx]
    
    for c in range(10):
        t = thresh_arr[c]
        if t > 0.0:
            mask = (pred == c) & (probs[c] < t)
            pred[mask] = fallback[mask]
    return pred

def main():
    print("Initializing Final Optimized Ensemble Inference Pipeline (No-TTA)...")
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    
    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Total Test Images: {len(test_images)}")
    
    # 1. Load Synergistic Model
    print("Loading Synergistic Model...")
    syn_model = FloodNetSynergisticNet(num_classes=10).to(device)
    syn_weights = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    if os.path.exists(syn_weights):
        state = torch.load(syn_weights, map_location=device, weights_only=True)
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
        syn_model.load_state_dict(state, strict=True)
    syn_model.eval()
    
    # 2. Load Meta-Stacked Model
    print("Loading Meta-Stacked Model...")
    meta_model = FloodNetCompetitiveModel(num_classes=10).to(device)
    meta_model.load_checkpoints(
        unet_path="model_checkpoint/FloodNet_UNet/best_unet_weights.pt",
        fcn_path="model_checkpoint/FloodNet_FCN/best_fcn_weights.pt",
        deeplab_path="model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt",
        meta_path="model_checkpoint/FloodNet_Meta/meta_layer_weights.pt",
        device=device
    )
    meta_model.eval()
    
    # Disable cudnn benchmark for stability on ROCm/AMD
    torch.backends.cudnn.benchmark = False
    
    # Load optimized parameters dynamically from config
    config_path = "model_checkpoint/ensemble_optimized_config.pt"
    if os.path.exists(config_path):
        print(f"Loading optimized config from {config_path}...")
        config = torch.load(config_path, map_location='cpu')
        best_w = np.array(config['best_w'], dtype=np.float32)
        best_t = np.array(config['best_t'], dtype=np.float32)
        best_area = config.get('best_area', 0)
        print("Loaded configurations successfully!")
    else:
        print("WARNING: Optimized config not found, using default fallback values.")
        best_w = np.array([0.5000, 0.6273, 0.5410, 0.5248, 0.8813, 1.0000, 0.4758, 1.0000, 0.6841, 0.4611], dtype=np.float32)
        best_t = np.array([0.9794, 0.5347, 0.0000, 0.5355, 0.0000, 0.0000, 0.0000, 0.0000, 0.4006, 0.0000], dtype=np.float32)
        best_area = 0
        
    print("Optimization Parameters:")
    print(f"  w_syn: {best_w.tolist()}")
    print(f"  thresh: {best_t.tolist()}")
    print(f"  min_area: {best_area}")
    
    if "DRY_RUN" in os.environ:
        print("[DRY RUN] Limiting inference to first 10 images.")
        test_images = test_images[:10]
        
    out_file = "final_optimized_ensemble_submission.csv"
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["IMG_ID", "EncodedString"])
        
        with torch.no_grad():
            for img_path in tqdm(test_images, desc="Running Inference"):
                filename = os.path.basename(img_path).replace('.jpg', '')
                base_img = cv2.imread(img_path)
                if base_img is None:
                    print(f"Warning: could not read {img_path}")
                    continue
                orig_h, orig_w = base_img.shape[:2]
                
                # Image processing
                img_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
                
                img_tensor = img_resized.astype(np.float32) / 255.0
                img_tensor = (img_tensor - mean) / std
                img_tensor = img_tensor.transpose(2, 0, 1)
                
                x = torch.tensor(img_tensor[None, ...], dtype=torch.float32).to(device)
                
                # Predict Synergistic Net (Standard Pass only)
                out_syn = syn_model(x)
                p_syn_main = F.softmax(out_syn['main_output'], dim=1).cpu().numpy()[0]
                p_syn_unet = F.softmax(out_syn['unet_output'], dim=1).cpu().numpy()[0]
                p_syn_dl = F.softmax(out_syn['deeplab_output'], dim=1).cpu().numpy()[0]
                
                p_syn = 0.4 * p_syn_main + 0.3 * p_syn_unet + 0.3 * p_syn_dl
                p_syn[0] = p_syn_main[0] # keep class 0 unscaled
                del p_syn_main, p_syn_unet, p_syn_dl, out_syn
                
                # Predict Meta Net (Standard Pass only)
                meta_logits = meta_model(x, use_tta=False)
                p_meta = F.softmax(meta_logits, dim=1).cpu().numpy()[0]
                del meta_logits
                
                # Blend using class-specific weights
                w_syn_reshaped = best_w.reshape(10, 1, 1)
                p_blend = w_syn_reshaped * p_syn + (1.0 - w_syn_reshaped) * p_meta
                p_blend[0] = p_syn[0] # class 0 unscaled
                
                # Apply multiclass thresholds
                pred_mask = apply_multiclass_thresholding(p_blend, best_t)
                
                # Apply morphological cleanup
                if best_area > 0:
                    pred_mask = neighbor_fill_cleanup(pred_mask.astype(np.uint8), min_area=best_area)
                
                # Resize final mask back to high resolution
                pred_hr = cv2.resize(pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
                classes_in_pred = set(np.unique(pred_hr))
                for class_id in range(10):
                    row_id = f"{filename}_{class_id:02d}"
                    if class_id not in classes_in_pred:
                        writer.writerow([row_id, ""])
                    else:
                        rle_str = mask2rle((pred_hr == class_id).astype(np.uint8))
                        writer.writerow([row_id, rle_str])
                        
                del x, p_syn, p_meta, p_blend, pred_mask, pred_hr
                gc.collect()
                torch.cuda.empty_cache()
                
    print(f"\nSUCCESS! Generated optimal No-TTA submission file: {out_file}")

if __name__ == '__main__':
    main()
