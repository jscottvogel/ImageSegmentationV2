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
from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader
import csv

from scipy.ndimage import distance_transform_edt
from synergistic_model import FloodNetSynergisticNet
from competitive_model import FloodNetCompetitiveModel
from optimized_pytorch_version import DatasetConfig
from scratch.eval_advanced_tta import neighbor_fill_cleanup

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

def mask2rle(img: np.ndarray) -> str:
    """Highly optimized 1-based RLE encoder."""
    pixels = img.T.ravel()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    runs[::2] += 1
    if len(runs) == 0:
        return ""
    return ' '.join(map(str, runs))

class TestDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path).replace('.jpg', '')
        
        base_img = cv2.imread(img_path)
        orig_h, orig_w = base_img.shape[:2]
        
        img_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
        
        img_tensor = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.transpose(2, 0, 1)
        
        return img_tensor, filename, orig_h, orig_w

def process_single_image(args):
    filename, orig_h, orig_w, p_syn_notta, p_meta, best_w, best_t, best_areas = args
    
    # 1. Blend
    best_w_arr = np.array(best_w).reshape(10, 1, 1)
    best_t_arr = np.array(best_t)
    
    p_blend = best_w_arr * p_syn_notta + (1.0 - best_w_arr) * p_meta
    p_blend[0] = p_syn_notta[0] # Class 0 unscaled
    
    # 2. Multiclass thresholding fallback logic
    pred_labels = np.argmax(p_blend, axis=0)
    
    fallback_classes = [c for c in range(10) if best_t_arr[c] == 0.0]
    if len(fallback_classes) == 0:
        fallback_classes = [2]
        
    fallback_probs = p_blend[fallback_classes]
    fallback_idx = np.argmax(fallback_probs, axis=0)
    fallback = np.array(fallback_classes)[fallback_idx]
    
    for c in range(10):
        t = best_t_arr[c]
        if t > 0.0:
            mask = (pred_labels == c) & (p_blend[c] < t)
            pred_labels[mask] = fallback[mask]
            
    # 3. Morphological area cleanup
    pred_labels = neighbor_fill_cleanup_class_specific(pred_labels, best_areas)
        
    # 4. Resize to high-res (nearest)
    pred_hr = cv2.resize(pred_labels, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    classes_present = set(np.unique(pred_labels))
    
    rows = []
    for class_id in range(10):
        if class_id not in classes_present:
            rows.append([f"{filename}_{class_id:02d}", ""])
        else:
            rle = mask2rle((pred_hr == class_id).astype(np.uint8))
            rows.append([f"{filename}_{class_id:02d}", rle])
            
    return rows

def main():
    print("Initializing final Kaggle-aligned submission pipeline...")
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
    
    # Load optimal hyperparameters from config
    config_path = "model_checkpoint/ensemble_kaggle_config.pt"
    if not os.path.exists(config_path):
        print(f"Error: optimized config not found at {config_path}!")
        return
        
    config = torch.load(config_path)
    best_w = config['best_w']
    best_t = config['best_t']
    best_areas = config.get('best_areas', [config.get('best_area', 96)] * 10)
    
    print("\nLoaded Optimal Hyperparameters:")
    print(f"  - Morphological min_areas: {best_areas}")
    print("  - Weights (w_syn):", [round(w, 4) for w in best_w])
    print("  - Thresholds:", [round(t, 4) for t in best_t])
    
    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    
    if len(test_images) == 0:
        print("Error: No test images found!")
        return
        
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
    
    # Disable cudnn benchmark for optimization stability
    torch.backends.cudnn.benchmark = False
    
    dataset = TestDataset(test_images)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    
    output_csv = "final_kaggle_submission.csv"
    
    # Initialize CPU process pool for postprocessing and RLE
    pool = Pool(processes=4)
    
    try:
        with open(output_csv, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["IMG_ID", "EncodedString"])
            
            with torch.no_grad():
                for batch in tqdm(loader, desc="Inference"):
                    img_tensors, filenames, orig_hs, orig_ws = batch
                    
                    x = img_tensors.to(device)
                    
                    # Predict Synergistic Net
                    out_syn = syn_model(x)
                    p_syn_main = F.softmax(out_syn['main_output'], dim=1).cpu().numpy()
                    p_syn_unet = F.softmax(out_syn['unet_output'], dim=1).cpu().numpy()
                    p_syn_dl = F.softmax(out_syn['deeplab_output'], dim=1).cpu().numpy()
                    
                    # Fuse synergistic heads
                    p_syn_notta = 0.4 * p_syn_main + 0.3 * p_syn_unet + 0.3 * p_syn_dl
                    for b in range(len(filenames)):
                        p_syn_notta[b, 0] = p_syn_main[b, 0]
                        
                    # Predict Meta-Stacked Net
                    meta_logits = meta_model(x)
                    p_meta = F.softmax(meta_logits, dim=1).cpu().numpy()
                    
                    # Package inputs for parallel CPU postprocessing
                    precomputed_inputs = []
                    for b in range(len(filenames)):
                        precomputed_inputs.append((
                            filenames[b], int(orig_hs[b]), int(orig_ws[b]),
                            p_syn_notta[b], p_meta[b],
                            best_w, best_t, best_areas
                        ))
                        
                    batch_results = pool.map(process_single_image, precomputed_inputs)
                    
                    for image_rows in batch_results:
                        writer.writerows(image_rows)
                        
                    del x, out_syn, meta_logits, p_syn_notta, p_meta
                    gc.collect()
                    torch.cuda.empty_cache()
                    
    finally:
        pool.close()
        pool.join()
        
    print(f"\nSUCCESS! Generated final optimized submission file: {output_csv}")

if __name__ == '__main__':
    main()
