import os
import argparse
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
from optimized_pytorch_version import DatasetConfig

def mask2rle(img: np.ndarray) -> str:
    """Highly optimized 1-based RLE encoder."""
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    runs[::2] += 1
    if len(runs) == 0:
        return ""
    return ' '.join(str(x) for x in runs)

def neighbor_fill_cleanup(pred_labels: np.ndarray, min_area=100) -> np.ndarray:
    if min_area <= 0:
        return pred_labels
    
    # 1. Identify small isolated noise components in classes 1-9
    noise_mask = np.zeros(pred_labels.shape, dtype=bool)
    for class_id in range(1, 10):
        class_mask = (pred_labels == class_id).astype(np.uint8)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for i in range(1, num_components):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                noise_mask[labels == i] = True
                
    if not np.any(noise_mask):
        return pred_labels
        
    # 2. Fill the noise mask pixels with the value of the nearest valid pixel label
    indices = distance_transform_edt(noise_mask, return_distances=False, return_indices=True)
    clean_labels = pred_labels[indices[0], indices[1]]
    return clean_labels

def multiscale_tta_inference(model, image_tensor):
    model.eval()
    with torch.no_grad():
        preds_dict = model(image_tensor)
        out_standard = preds_dict['main_output']
        out_standard_probs = F.softmax(out_standard, dim=1).cpu()
        
        # TTA Horizontal Flip Pass
        img_flipped = torch.flip(image_tensor, dims=[3])
        preds_dict_flipped = model(img_flipped)
        out_flipped = preds_dict_flipped['main_output']
        out_flipped_probs = F.softmax(out_flipped, dim=1).cpu()
        out_unflipped_probs = torch.flip(out_flipped_probs, dims=[3])
        
        fused_probs = (out_standard_probs + out_unflipped_probs) * 0.5
        
    return fused_probs.squeeze(0).numpy()

def main():
    print(f"Starting Calibrated Inference Pipeline with Neighbor-Fill Noise Cleanup (T90, T95, T99)...")
    
    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    
    if len(test_images) == 0:
        print("CRITICAL ERROR: No test images found!")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Total Test Targets: {len(test_images)}")
    
    # Load model
    model = FloodNetSynergisticNet(num_classes=10).to(device)
    weights_path = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    if os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        state = torch.load(weights_path, map_location=device, weights_only=True)
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
        model.load_state_dict(state)
    else:
        print("CRITICAL ERROR: weights not found!")
        return
    model.eval()
    
    submission_t90_data = []
    submission_t95_data = []
    submission_t99_data = []
    
    for idx, img_path in enumerate(tqdm(test_images, desc="Inference + EDT Cleanup")):
        filename = os.path.basename(img_path).replace('.jpg', '')
        base_img = cv2.imread(img_path)
        orig_h, orig_w = base_img.shape[:2]
        
        # Preprocessing
        img_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
        
        img_tensor = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_tensor = (img_tensor - mean) / std
        img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            fused_probs = multiscale_tta_inference(model, img_tensor)
            
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(fused_probs[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # Common fallback
            fallback = np.argmax(probs_resized[1:], axis=0) + 1
            pred_base = np.argmax(probs_resized, axis=0).astype(np.uint8)
            c0_mask = (pred_base == 0)
            
            # --- THRESHOLD 0.90 + Cleanup ---
            pred_t90 = pred_base.copy()
            low_conf_t90 = c0_mask & (probs_resized[0] < 0.90)
            if np.any(low_conf_t90):
                pred_t90[low_conf_t90] = fallback[low_conf_t90].astype(np.uint8)
            pred_t90 = neighbor_fill_cleanup(pred_t90, min_area=100)
            
            # --- THRESHOLD 0.95 + Cleanup ---
            pred_t95 = pred_base.copy()
            low_conf_t95 = c0_mask & (probs_resized[0] < 0.95)
            if np.any(low_conf_t95):
                pred_t95[low_conf_t95] = fallback[low_conf_t95].astype(np.uint8)
            pred_t95 = neighbor_fill_cleanup(pred_t95, min_area=100)
            
            # --- THRESHOLD 0.99 + Cleanup ---
            pred_t99 = pred_base.copy()
            low_conf_t99 = c0_mask & (probs_resized[0] < 0.99)
            if np.any(low_conf_t99):
                pred_t99[low_conf_t99] = fallback[low_conf_t99].astype(np.uint8)
            pred_t99 = neighbor_fill_cleanup(pred_t99, min_area=100)
            
        # RLE encoding
        present_t90 = set(np.unique(pred_t90))
        present_t95 = set(np.unique(pred_t95))
        present_t99 = set(np.unique(pred_t99))
        
        for class_id in range(10):
            # t90
            if class_id not in present_t90:
                enc_t90 = ""
            else:
                enc_t90 = mask2rle((pred_t90 == class_id).astype(np.uint8))
            submission_t90_data.append([f"{filename}_{class_id:02d}", enc_t90])
            
            # t95
            if class_id not in present_t95:
                enc_t95 = ""
            else:
                enc_t95 = mask2rle((pred_t95 == class_id).astype(np.uint8))
            submission_t95_data.append([f"{filename}_{class_id:02d}", enc_t95])
            
            # t99
            if class_id not in present_t99:
                enc_t99 = ""
            else:
                enc_t99 = mask2rle((pred_t99 == class_id).astype(np.uint8))
            submission_t99_data.append([f"{filename}_{class_id:02d}", enc_t99])
            
        del img_tensor, fused_probs, probs_resized, pred_t90, pred_t95, pred_t99
        gc.collect()
        torch.cuda.empty_cache()
        
    out_t90 = 'synergistic_neighbor_filled_t90_submission.csv'
    out_t95 = 'synergistic_neighbor_filled_t95_submission.csv'
    out_t99 = 'synergistic_neighbor_filled_t99_submission.csv'
    
    pd.DataFrame(submission_t90_data, columns=["IMG_ID", "EncodedString"]).to_csv(out_t90, index=False)
    pd.DataFrame(submission_t95_data, columns=["IMG_ID", "EncodedString"]).to_csv(out_t95, index=False)
    pd.DataFrame(submission_t99_data, columns=["IMG_ID", "EncodedString"]).to_csv(out_t99, index=False)
    
    print(f"SUCCESS! Neighbor-filled submissions saved to:\n  - {out_t90}\n  - {out_t95}\n  - {out_t99}")

if __name__ == '__main__':
    main()
