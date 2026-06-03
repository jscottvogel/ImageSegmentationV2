import os
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
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    runs[::2] += 1
    if len(runs) == 0:
        return ""
    return ' '.join(str(x) for x in runs)

# High-resolution connected components morph clean
def neighbor_fill_cleanup(pred_labels: np.ndarray, min_area=8) -> np.ndarray:
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    print(f"Found {len(test_images)} test images.")
    
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
    
    # Containers
    sub_syn_t95_a25 = []
    sub_syn_t90_a8 = []
    sub_hybrid_t95_a25 = []
    
    # Preprocessing constants
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    torch.backends.cudnn.benchmark = False
    
    with torch.no_grad():
        for idx, img_path in enumerate(tqdm(test_images, desc="Running Pure TTA Pipelines")):
            filename = os.path.basename(img_path).replace('.jpg', '')
            img_bgr = cv2.imread(img_path)
            orig_h, orig_w = img_bgr.shape[:2]
            
            # Preprocess
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_tensor = img_resized.astype(np.float32) / 255.0
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            img_flipped = torch.flip(img_tensor, dims=[3])
            
            # --- 1. Synergistic Model Inference ---
            out_syn_std = syn_model(img_tensor)
            out_syn_flip = syn_model(img_flipped)
            
            p_syn_main_std = F.softmax(out_syn_std['main_output'], dim=1)
            p_syn_main_flip = F.softmax(out_syn_flip['main_output'], dim=1)
            p_syn_main = (p_syn_main_std + torch.flip(p_syn_main_flip, dims=[3])) * 0.5
            
            p_syn_unet_std = F.softmax(out_syn_std['unet_output'], dim=1)
            p_syn_unet_flip = F.softmax(out_syn_flip['unet_output'], dim=1)
            p_syn_unet = (p_syn_unet_std + torch.flip(p_syn_unet_flip, dims=[3])) * 0.5
            
            p_syn_dl_std = F.softmax(out_syn_std['deeplab_output'], dim=1)
            p_syn_dl_flip = F.softmax(out_syn_flip['deeplab_output'], dim=1)
            p_syn_dl = (p_syn_dl_std + torch.flip(p_syn_dl_flip, dims=[3])) * 0.5
            
            # Optimal decoder blending from validation
            p_syn_blend = 0.4 * p_syn_main + 0.3 * p_syn_unet + 0.3 * p_syn_dl
            p_syn_blend[:, 0] = p_syn_main[:, 0] # Keep unscaled background probability
            p_syn_np = p_syn_blend.cpu().squeeze(0).numpy()
            
            # --- 2. Hybrid Model Inference ---
            out_u_std = unet(img_tensor)
            out_u_flip = unet(img_flipped)
            p_unet = (F.softmax(out_u_std['main_output'], dim=1) + torch.flip(F.softmax(out_u_flip['main_output'], dim=1), dims=[3])) * 0.5
            p_unet_np = p_unet.cpu().squeeze(0).numpy()
            
            out_f_std = fcn(img_tensor)
            out_f_flip = fcn(img_flipped)
            p_fcn = (F.softmax(out_f_std['main_output'], dim=1) + torch.flip(F.softmax(out_f_flip['main_output'], dim=1), dims=[3])) * 0.5
            p_fcn_np = p_fcn.cpu().squeeze(0).numpy()
            
            out_d_std = deeplab(img_tensor)
            out_d_flip = deeplab(img_flipped)
            p_dl = (F.softmax(out_d_std['main_output'], dim=1) + torch.flip(F.softmax(out_d_flip['main_output'], dim=1), dims=[3])) * 0.5
            p_dl_np = p_dl.cpu().squeeze(0).numpy()
            
            # Class-weighted hybrid blending
            p_hybrid_np = (p_dl_np * w_dl + p_unet_np * w_unet + p_fcn_np * w_fcn) / total_w
            
            # --- Resize and Post-process predictions ---
            probs_syn_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            probs_hybrid_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_syn_resized[c] = cv2.resize(p_syn_np[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                probs_hybrid_resized[c] = cv2.resize(p_hybrid_np[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                
            # Fallbacks and basic argmax
            fallback_syn = np.argmax(probs_syn_resized[1:], axis=0) + 1
            pred_syn = np.argmax(probs_syn_resized, axis=0).astype(np.uint8)
            
            fallback_hybrid = np.argmax(probs_hybrid_resized[1:], axis=0) + 1
            pred_hybrid = np.argmax(probs_hybrid_resized, axis=0).astype(np.uint8)
            
            # Configuration A: Synergistic TTA + T=0.95 + Area=25
            pred_a = pred_syn.copy()
            c0_mask_a = (pred_a == 0)
            low_conf_a = c0_mask_a & (probs_syn_resized[0] < 0.95)
            if np.any(low_conf_a):
                pred_a[low_conf_a] = fallback_syn[low_conf_a].astype(np.uint8)
            pred_a = neighbor_fill_cleanup(pred_a, min_area=25)
            
            # Configuration B: Synergistic TTA + T=0.90 + Area=8
            pred_b = pred_syn.copy()
            c0_mask_b = (pred_b == 0)
            low_conf_b = c0_mask_b & (probs_syn_resized[0] < 0.90)
            if np.any(low_conf_b):
                pred_b[low_conf_b] = fallback_syn[low_conf_b].astype(np.uint8)
            pred_b = neighbor_fill_cleanup(pred_b, min_area=8)
            
            # Configuration C: Hybrid TTA + T=0.95 + Area=25
            pred_c = pred_hybrid.copy()
            c0_mask_c = (pred_c == 0)
            low_conf_c = c0_mask_c & (probs_hybrid_resized[0] < 0.95)
            if np.any(low_conf_c):
                pred_c[low_conf_c] = fallback_hybrid[low_conf_c].astype(np.uint8)
            pred_c = neighbor_fill_cleanup(pred_c, min_area=25)
            
            # RLE encode
            classes_a = set(np.unique(pred_a))
            classes_b = set(np.unique(pred_b))
            classes_c = set(np.unique(pred_c))
            
            for class_id in range(10):
                # A
                enc_a = mask2rle((pred_a == class_id).astype(np.uint8)) if class_id in classes_a else ""
                sub_syn_t95_a25.append([f"{filename}_{class_id:02d}", enc_a])
                # B
                enc_b = mask2rle((pred_b == class_id).astype(np.uint8)) if class_id in classes_b else ""
                sub_syn_t90_a8.append([f"{filename}_{class_id:02d}", enc_b])
                # C
                enc_c = mask2rle((pred_c == class_id).astype(np.uint8)) if class_id in classes_c else ""
                sub_hybrid_t95_a25.append([f"{filename}_{class_id:02d}", enc_c])
                
            del img_tensor, img_flipped, pred_a, pred_b, pred_c
            gc.collect()
            torch.cuda.empty_cache()
            
    # Save CSVs
    pd.DataFrame(sub_syn_t95_a25, columns=["IMG_ID", "EncodedString"]).to_csv("synergistic_pure_tta_t95_area25_submission.csv", index=False)
    print("Saved synergistic_pure_tta_t95_area25_submission.csv")
    
    pd.DataFrame(sub_syn_t90_a8, columns=["IMG_ID", "EncodedString"]).to_csv("synergistic_pure_tta_t90_area8_submission.csv", index=False)
    print("Saved synergistic_pure_tta_t90_area8_submission.csv")
    
    pd.DataFrame(sub_hybrid_t95_a25, columns=["IMG_ID", "EncodedString"]).to_csv("hybrid_pure_tta_t95_area25_submission.csv", index=False)
    print("Saved hybrid_pure_tta_t95_area25_submission.csv")

if __name__ == '__main__':
    main()
