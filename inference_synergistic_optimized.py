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

def get_optimized_probabilities(model, image_tensor):
    model.eval()
    with torch.no_grad():
        preds_dict = model(image_tensor)
        
        p_main = F.softmax(preds_dict['main_output'], dim=1)
        p_unet = F.softmax(preds_dict['unet_output'], dim=1)
        p_deeplab = F.softmax(preds_dict['deeplab_output'], dim=1)
        
        # Blending of spatial fusion block + individual decoders (Optimal config from validation sweep)
        p_blend = 0.4 * p_main + 0.3 * p_unet + 0.3 * p_deeplab
        # Preserve unscaled Class 0 from the main fused output
        p_blend[:, 0] = p_main[:, 0]
        
    return p_blend.squeeze(0).cpu().numpy()

def main():
    print(f"Starting Optimized Inference Pipeline (Blending, No-TTA, Neighbor-Fill Cleanup)...")
    
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
        model.load_state_dict(state, strict=False)
    else:
        print("CRITICAL ERROR: weights not found!")
        return
    model.eval()
    
    submission_t95_data = []
    submission_t99_data = []
    submission_t995_data = []
    
    for idx, img_path in enumerate(tqdm(test_images, desc="Optimized Inference")):
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
            fused_probs = get_optimized_probabilities(model, img_tensor)
            
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(fused_probs[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            # Common fallback
            fallback = np.argmax(probs_resized[1:], axis=0) + 1
            pred_base = np.argmax(probs_resized, axis=0).astype(np.uint8)
            c0_mask = (pred_base == 0)
            
            # --- THRESHOLD 0.95 + Cleanup ---
            pred_t95 = pred_base.copy()
            low_conf_t95 = c0_mask & (probs_resized[0] < 0.95)
            if np.any(low_conf_t95):
                pred_t95[low_conf_t95] = fallback[low_conf_t95].astype(np.uint8)
            pred_t95 = neighbor_fill_cleanup(pred_t95, min_area=120)
            
            # --- THRESHOLD 0.99 + Cleanup ---
            pred_t99 = pred_base.copy()
            low_conf_t99 = c0_mask & (probs_resized[0] < 0.99)
            if np.any(low_conf_t99):
                pred_t99[low_conf_t99] = fallback[low_conf_t99].astype(np.uint8)
            pred_t99 = neighbor_fill_cleanup(pred_t99, min_area=120)
            
            # --- THRESHOLD 0.995 + Cleanup ---
            pred_t995 = pred_base.copy()
            low_conf_t995 = c0_mask & (probs_resized[0] < 0.995)
            if np.any(low_conf_t995):
                pred_t995[low_conf_t995] = fallback[low_conf_t995].astype(np.uint8)
            pred_t995 = neighbor_fill_cleanup(pred_t995, min_area=120)
            
        # RLE encoding
        present_t95 = set(np.unique(pred_t95))
        present_t99 = set(np.unique(pred_t99))
        present_t995 = set(np.unique(pred_t995))
        
        for class_id in range(10):
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
            
            # t995
            if class_id not in present_t995:
                enc_t995 = ""
            else:
                enc_t995 = mask2rle((pred_t995 == class_id).astype(np.uint8))
            submission_t995_data.append([f"{filename}_{class_id:02d}", enc_t995])
            
        del img_tensor, fused_probs, probs_resized, pred_t95, pred_t99, pred_t995
        gc.collect()
        torch.cuda.empty_cache()
        
    out_t95 = 'synergistic_optimized_t95_submission.csv'
    out_t99 = 'synergistic_optimized_t99_submission.csv'
    out_t995 = 'synergistic_optimized_t995_submission.csv'
    
    pd.DataFrame(submission_t95_data, columns=["IMG_ID", "EncodedString"]).to_csv(out_t95, index=False)
    pd.DataFrame(submission_t99_data, columns=["IMG_ID", "EncodedString"]).to_csv(out_t99, index=False)
    pd.DataFrame(submission_t995_data, columns=["IMG_ID", "EncodedString"]).to_csv(out_t995, index=False)
    
    print(f"SUCCESS! Optimized submissions saved to:\n  - {out_t95}\n  - {out_t99}\n  - {out_t995}")

if __name__ == '__main__':
    main()
