import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"
import glob
import cv2
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import gc
from tqdm import tqdm

from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig
from unet_version import StandardUNet
from fcn_version import ResNet50FCN

# Clean 1-based RLE encoder (Kaggle compliant)
def mask2rle(img: np.ndarray) -> str:
    pixels = img.T.ravel()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    runs[::2] += 1
    if len(runs) == 0:
        return ""
    return ' '.join(map(str, runs))

# Microscopic false-positive cleanup
def safe_morphological_cleanup(pred_labels: np.ndarray, min_area=50) -> np.ndarray:
    clean_labels = pred_labels.copy()
    for class_id in np.unique(pred_labels):
        if class_id == 0: continue
        class_mask = (pred_labels == class_id).astype(np.uint8)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for i in range(1, num_components):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                clean_labels[labels == i] = 0
    return clean_labels

def normalize_imagenet(img_tensor, device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (img_tensor - mean) / std

def load_weights_custom(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if 'n_averaged' in state:
        del state['n_averaged']
    clean_state = {}
    for k, v in state.items():
        k = k.replace('module.', '').replace('_orig_mod.', '')
        clean_state[k] = v
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference pipeline on: {device}")
    
    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    
    if len(test_images) == 0:
        print("CRITICAL ERROR: No test images found!")
        return
        
    print(f"Found {len(test_images)} test images.")
    
    if "DRY_RUN" in os.environ:
        print("[DRY RUN] Limiting inference to first 10 images.")
        test_images = test_images[:10]
        
    # 1. Load All Models
    print("Loading DeepLabV3Plus...")
    deeplab = CustomDeepLabV3Plus(num_classes=10).to(device)
    deeplab = load_weights_custom(deeplab, "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt", device)
    
    print("Loading UNet...")
    unet = StandardUNet(num_classes=10).to(device)
    unet = load_weights_custom(unet, "model_checkpoint/FloodNet_UNet/best_unet_weights.pt", device)
    
    print("Loading FCN...")
    fcn = ResNet50FCN(num_classes=10).to(device)
    fcn = load_weights_custom(fcn, "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt", device)
    
    # Load Meta layer if exists
    meta_layer = None
    meta_weights_path = "model_checkpoint/FloodNet_Meta/meta_layer_weights.pt"
    if os.path.exists(meta_weights_path):
        print("Loading Meta Stacked Layer...")
        meta_state = torch.load(meta_weights_path, map_location=device, weights_only=True)
        meta_layer = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=1).to(device)
        meta_layer.load_state_dict(meta_state)
        meta_layer.eval()
    else:
        print("WARNING: Meta layer weights not found. Skipping Meta-Learner Stacked Ensemble.")

    # 3. Ensemble Class Weights
    w_dl = np.array([1.1350, 0.0000, 0.0000, 0.0000, 0.0000, 2.5432, 0.0000, 0.0000, 0.9120, 0.0000], dtype=np.float32).reshape(10, 1, 1)
    w_unet = np.array([0.0000, 0.9893, 2.2479, 0.7506, 1.2237, 1.8659, 2.7888, 1.9066, 0.9812, 1.4061], dtype=np.float32).reshape(10, 1, 1)
    w_fcn = np.array([1.8123, 0.8671, 1.3632, 1.2473, 0.9071, 1.0935, 0.8761, 0.6374, 1.0507, 0.9598], dtype=np.float32).reshape(10, 1, 1)
    total_w = w_dl + w_unet + w_fcn

    # Disable benchmark mode for ROCm/CUDA stability
    torch.backends.cudnn.benchmark = False
    
    out_dl = "deeplabv3plus_best_submission.csv"
    out_unet = "unet_best_submission.csv"
    out_fcn = "fcn_best_submission.csv"
    out_hybrid = "hybrid_ensemble_submission.csv"
    out_meta = "meta_ensemble_submission.csv"
    
    import csv
    with open(out_dl, 'w', newline='') as f_dl, \
         open(out_unet, 'w', newline='') as f_unet, \
         open(out_fcn, 'w', newline='') as f_fcn, \
         open(out_hybrid, 'w', newline='') as f_hybrid, \
         (open(out_meta, 'w', newline='') if meta_layer is not None else open(os.devnull, 'w')) as f_meta:
         
        w_dl_csv = csv.writer(f_dl)
        w_unet_csv = csv.writer(f_unet)
        w_fcn_csv = csv.writer(f_fcn)
        w_hybrid_csv = csv.writer(f_hybrid)
        w_meta_csv = csv.writer(f_meta) if meta_layer is not None else None
        
        w_dl_csv.writerow(["IMG_ID", "EncodedString"])
        w_unet_csv.writerow(["IMG_ID", "EncodedString"])
        w_fcn_csv.writerow(["IMG_ID", "EncodedString"])
        w_hybrid_csv.writerow(["IMG_ID", "EncodedString"])
        if w_meta_csv is not None:
            w_meta_csv.writerow(["IMG_ID", "EncodedString"])
            
        # 4. Processing Loop
        for img_path in tqdm(test_images, desc="Running Unified Ensemble Pipeline"):
            filename = os.path.basename(img_path).replace('.jpg', '')
            
            # Load and preprocess
            img_bgr = cv2.imread(img_path)
            orig_h, orig_w = img_bgr.shape[:2]
            
            base_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(base_img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
            
            img_tensor = torch.tensor(resized_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            norm_img = normalize_imagenet(img_tensor, device)
            
            # Forward pass 1: DeepLab
            out_dl = deeplab(norm_img)
            p_dl = F.softmax(out_dl['main_output'], dim=1).squeeze(0).cpu().numpy()
            del out_dl
            
            # Forward pass 2: UNet
            out_unet = unet(norm_img)
            if isinstance(out_unet, dict):
                out_unet = out_unet['main_output']
            p_unet = F.softmax(out_unet, dim=1).squeeze(0).cpu().numpy()
            del out_unet
            
            # Forward pass 3: FCN
            out_fcn = fcn(norm_img)
            if isinstance(out_fcn, dict):
                out_fcn = out_fcn['main_output']
            p_fcn = F.softmax(out_fcn, dim=1).squeeze(0).cpu().numpy()
            del out_fcn
            
            # Calculate Hybrid probabilities
            p_hybrid = (p_dl * w_dl + p_unet * w_unet + p_fcn * w_fcn) / total_w
            
            # Calculate Meta probabilities
            p_meta = None
            if meta_layer is not None:
                stacked = np.concatenate([p_unet, p_fcn, p_dl], axis=0) # order: UNet, FCN, DeepLab
                stacked_tensor = torch.tensor(stacked[None, ...], dtype=torch.float32).to(device)
                meta_logits = meta_layer(stacked_tensor)
                p_meta = F.softmax(meta_logits, dim=1).squeeze(0).cpu().numpy()
                del stacked, stacked_tensor, meta_logits
                
            # A. Process DeepLab
            p_dl_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                p_dl_resized[c] = cv2.resize(p_dl[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            labels_dl = np.argmax(p_dl_resized, axis=0).astype(np.uint8)
            del p_dl_resized, p_dl
            labels_dl = safe_morphological_cleanup(labels_dl, min_area=50)
            for class_id in range(10):
                encoded_dl = mask2rle((labels_dl == class_id).astype(np.uint8))
                w_dl_csv.writerow([f"{filename}_{class_id:02d}", encoded_dl])
            del labels_dl
            
            # B. Process UNet
            p_unet_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                p_unet_resized[c] = cv2.resize(p_unet[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            labels_unet = np.argmax(p_unet_resized, axis=0).astype(np.uint8)
            del p_unet_resized, p_unet
            labels_unet = safe_morphological_cleanup(labels_unet, min_area=50)
            for class_id in range(10):
                encoded_unet = mask2rle((labels_unet == class_id).astype(np.uint8))
                w_unet_csv.writerow([f"{filename}_{class_id:02d}", encoded_unet])
            del labels_unet
            
            # C. Process FCN
            p_fcn_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                p_fcn_resized[c] = cv2.resize(p_fcn[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            labels_fcn = np.argmax(p_fcn_resized, axis=0).astype(np.uint8)
            del p_fcn_resized, p_fcn
            labels_fcn = safe_morphological_cleanup(labels_fcn, min_area=50)
            for class_id in range(10):
                encoded_fcn = mask2rle((labels_fcn == class_id).astype(np.uint8))
                w_fcn_csv.writerow([f"{filename}_{class_id:02d}", encoded_fcn])
            del labels_fcn
            
            # D. Process Hybrid Ensemble
            p_hybrid_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                p_hybrid_resized[c] = cv2.resize(p_hybrid[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            labels_hybrid = np.argmax(p_hybrid_resized, axis=0).astype(np.uint8)
            del p_hybrid_resized, p_hybrid
            labels_hybrid = safe_morphological_cleanup(labels_hybrid, min_area=50)
            for class_id in range(10):
                encoded_hybrid = mask2rle((labels_hybrid == class_id).astype(np.uint8))
                w_hybrid_csv.writerow([f"{filename}_{class_id:02d}", encoded_hybrid])
            del labels_hybrid
            
            # E. Process Meta Ensemble
            if meta_layer is not None:
                p_meta_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
                for c in range(10):
                    p_meta_resized[c] = cv2.resize(p_meta[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                labels_meta = np.argmax(p_meta_resized, axis=0).astype(np.uint8)
                del p_meta_resized, p_meta
                labels_meta = safe_morphological_cleanup(labels_meta, min_area=50)
                for class_id in range(10):
                    encoded_meta = mask2rle((labels_meta == class_id).astype(np.uint8))
                    w_meta_csv.writerow([f"{filename}_{class_id:02d}", encoded_meta])
                del labels_meta
                
            # Clean VRAM variables
            del img_tensor, norm_img
            gc.collect()
            torch.cuda.empty_cache()
            
    print("\nALL SUBMISSIONS GENERATED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
