import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import gc
from scipy.ndimage import distance_transform_edt

from synergistic_model import FloodNetSynergisticNet
from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig
from unet_version import StandardUNet
from fcn_version import ResNet50FCN

# Kaggle-compliant 1-based RLE encoder
def mask2rle(img: np.ndarray) -> str:
    pixels = img.T.ravel()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    runs[::2] += 1
    if len(runs) == 0:
        return ""
    return ' '.join(map(str, runs))

# Connected components morph clean
def neighbor_fill_cleanup(pred_labels: np.ndarray, min_area=64) -> np.ndarray:
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    print(f"Found {len(test_images)} test images.")
    
    if "DRY_RUN" in os.environ:
        print("[DRY RUN] Limiting inference to first 10 images.")
        test_images = test_images[:10]
        
    # 1. Load Synergistic Model
    print("Loading Synergistic Net...")
    syn_model = FloodNetSynergisticNet(num_classes=10).to(device)
    syn_model = load_weights_custom(syn_model, "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt", device)
    
    # 2. Load Base Models for Hybrid Ensemble
    print("Loading base models (DeepLab, UNet, FCN) for Hybrid Ensemble...")
    deeplab = CustomDeepLabV3Plus(num_classes=10).to(device)
    deeplab = load_weights_custom(deeplab, "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt", device)
    
    unet = StandardUNet(num_classes=10).to(device)
    unet = load_weights_custom(unet, "model_checkpoint/FloodNet_UNet/best_unet_weights.pt", device)
    
    fcn = ResNet50FCN(num_classes=10).to(device)
    fcn = load_weights_custom(fcn, "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt", device)
    
    # Hybrid weights
    w_dl = np.array([1.1350, 0.0000, 0.0000, 0.0000, 0.0000, 2.5432, 0.0000, 0.0000, 0.9120, 0.0000], dtype=np.float32).reshape(10, 1, 1)
    w_unet = np.array([0.0000, 0.9893, 2.2479, 0.7506, 1.2237, 1.8659, 2.7888, 1.9066, 0.9812, 1.4061], dtype=np.float32).reshape(10, 1, 1)
    w_fcn = np.array([1.8123, 0.8671, 1.3632, 1.2473, 0.9071, 1.0935, 0.8761, 0.6374, 1.0507, 0.9598], dtype=np.float32).reshape(10, 1, 1)
    total_w = w_dl + w_unet + w_fcn
    
    # Preprocessing constants
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    torch.backends.cudnn.benchmark = False
    
    out_syn_c0_c3 = "synergistic_tta_suppress_c0_c3_submission.csv"
    out_hybrid_c0_c3 = "hybrid_tta_suppress_c0_c3_submission.csv"
    out_hybrid_c0_c3_c1 = "hybrid_tta_suppress_c0_c3_c1_submission.csv"
    
    import csv
    with open(out_syn_c0_c3, 'w', newline='') as f_syn, \
         open(out_hybrid_c0_c3, 'w', newline='') as f_hybrid, \
         open(out_hybrid_c0_c3_c1, 'w', newline='') as f_hybrid_c1:
         
        w_syn = csv.writer(f_syn)
        w_hybrid = csv.writer(f_hybrid)
        w_hybrid_c1 = csv.writer(f_hybrid_c1)
        
        w_syn.writerow(["IMG_ID", "EncodedString"])
        w_hybrid.writerow(["IMG_ID", "EncodedString"])
        w_hybrid_c1.writerow(["IMG_ID", "EncodedString"])
        
        with torch.no_grad():
            for idx, img_path in enumerate(tqdm(test_images, desc="Running Final Inference Pipelines")):
                filename = os.path.basename(img_path).replace('.jpg', '')
                img_bgr = cv2.imread(img_path)
                orig_h, orig_w = img_bgr.shape[:2] # 480x640
                
                # Preprocess
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_normalized = img_rgb.astype(np.float32) / 255.0
                img_normalized = (img_normalized - mean) / std
                img_tensor_cpu = torch.tensor(img_normalized.transpose(2,0,1)[None, ...], dtype=torch.float32)
                img_flipped = torch.flip(img_tensor_cpu, dims=[3]).to(device)
                img_tensor = img_tensor_cpu.to(device)
                
                # --- 1. Synergistic Model TTA Inference ---
                out_syn_std = syn_model(img_tensor)
                out_syn_flip = syn_model(img_flipped)
                
                p_syn_main_std = F.softmax(out_syn_std['main_output'], dim=1).cpu()
                p_syn_main_flip = F.softmax(out_syn_flip['main_output'], dim=1).cpu()
                p_syn_main = (p_syn_main_std + torch.flip(p_syn_main_flip, dims=[3])) * 0.5
                
                p_syn_unet_std = F.softmax(out_syn_std['unet_output'], dim=1).cpu()
                p_syn_unet_flip = F.softmax(out_syn_flip['unet_output'], dim=1).cpu()
                p_syn_unet = (p_syn_unet_std + torch.flip(p_syn_unet_flip, dims=[3])) * 0.5
                
                p_syn_dl_std = F.softmax(out_syn_std['deeplab_output'], dim=1).cpu()
                p_syn_dl_flip = F.softmax(out_syn_flip['deeplab_output'], dim=1).cpu()
                p_syn_dl = (p_syn_dl_std + torch.flip(p_syn_dl_flip, dims=[3])) * 0.5
                
                p_syn_blend = 0.4 * p_syn_main + 0.3 * p_syn_unet + 0.3 * p_syn_dl
                p_syn_blend[:, 0] = p_syn_main[:, 0]
                p_syn_np = p_syn_blend.squeeze(0).numpy()
                
                # --- 2. Hybrid Model TTA Inference ---
                out_u_std = unet(img_tensor)
                out_u_flip = unet(img_flipped)
                p_unet = (F.softmax(out_u_std['main_output'], dim=1).cpu() + torch.flip(F.softmax(out_u_flip['main_output'], dim=1).cpu(), dims=[3])) * 0.5
                p_unet_np = p_unet.squeeze(0).numpy()
                
                out_f_std = fcn(img_tensor)
                out_f_flip = fcn(img_flipped)
                p_fcn = (F.softmax(out_f_std['main_output'], dim=1).cpu() + torch.flip(F.softmax(out_f_flip['main_output'], dim=1).cpu(), dims=[3])) * 0.5
                p_fcn_np = p_fcn.squeeze(0).numpy()
                
                out_d_std = deeplab(img_tensor)
                out_d_flip = deeplab(img_flipped)
                p_dl = (F.softmax(out_d_std['main_output'], dim=1).cpu() + torch.flip(F.softmax(out_d_flip['main_output'], dim=1).cpu(), dims=[3])) * 0.5
                p_dl_np = p_dl.squeeze(0).numpy()
                
                p_hybrid_np = (p_dl_np * w_dl + p_unet_np * w_unet + p_fcn_np * w_fcn) / total_w
                
                # --- 3. Post-Process & Threshold configurations ---
                # Configuration 1: Synergistic TTA + Complete Background Suppression (T=1.0) + Class-3 Suppression (T=0.50) + Area=64
                pred_1 = apply_multiclass_thresholding(p_syn_np, {0: 1.0, 3: 0.50})
                pred_1 = neighbor_fill_cleanup(pred_1.astype(np.uint8), min_area=64)
                
                # Configuration 2: Hybrid TTA + Complete Background Suppression (T=1.0) + Class-3 Suppression (T=0.50) + Area=64
                pred_2 = apply_multiclass_thresholding(p_hybrid_np, {0: 1.0, 3: 0.50})
                pred_2 = neighbor_fill_cleanup(pred_2.astype(np.uint8), min_area=64)
                
                # Configuration 3: Hybrid TTA + Complete Background Suppression (T=1.0) + Class-3 Suppression (T=0.50) + Class-1 Suppression (T=0.50) + Area=64
                pred_3 = apply_multiclass_thresholding(p_hybrid_np, {0: 1.0, 3: 0.50, 1: 0.50})
                pred_3 = neighbor_fill_cleanup(pred_3.astype(np.uint8), min_area=64)
                
                # RLE encoding (at 480x640 native)
                classes_1 = set(np.unique(pred_1))
                classes_2 = set(np.unique(pred_2))
                classes_3 = set(np.unique(pred_3))
                
                for class_id in range(10):
                    # 1
                    enc_1 = mask2rle((pred_1 == class_id).astype(np.uint8)) if class_id in classes_1 else ""
                    w_syn.writerow([f"{filename}_{class_id:02d}", enc_1])
                    # 2
                    enc_2 = mask2rle((pred_2 == class_id).astype(np.uint8)) if class_id in classes_2 else ""
                    w_hybrid.writerow([f"{filename}_{class_id:02d}", enc_2])
                    # 3
                    enc_3 = mask2rle((pred_3 == class_id).astype(np.uint8)) if class_id in classes_3 else ""
                    w_hybrid_c1.writerow([f"{filename}_{class_id:02d}", enc_3])
                    
                del img_tensor, img_flipped, pred_1, pred_2, pred_3
                gc.collect()
                torch.cuda.empty_cache()
                
    print(f"Saved {out_syn_c0_c3}")
    print(f"Saved {out_hybrid_c0_c3}")
    print(f"Saved {out_hybrid_c0_c3_c1}")

if __name__ == '__main__':
    main()
