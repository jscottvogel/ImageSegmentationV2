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
from torch.utils.data import Dataset, DataLoader

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
        
        return img_tensor, img_resized, filename, orig_h, orig_w

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
    print("Initializing Memory-Safe Batched Ensemble Inference Pipeline (No-TTA)...")
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
    
    if "DRY_RUN" in os.environ:
        print("[DRY RUN] Limiting inference to first 10 images.")
        test_images = test_images[:10]
        
    out_c1 = "ensemble_w50_t95_c3t50_c1t50_area64_submission.csv"
    out_c2 = "ensemble_w50_t100_c3t50_c1t50_area64_submission.csv"
    out_c3 = "ensemble_w50_t100_c3t50_c1t50_area128_submission.csv"
    
    import csv
    with open(out_c1, 'w', newline='') as f1, open(out_c2, 'w', newline='') as f2, open(out_c3, 'w', newline='') as f3:
        w1 = csv.writer(f1)
        w2 = csv.writer(f2)
        w3 = csv.writer(f3)
        w1.writerow(["IMG_ID", "EncodedString"])
        w2.writerow(["IMG_ID", "EncodedString"])
        w3.writerow(["IMG_ID", "EncodedString"])
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        with torch.no_grad():
            for img_path in tqdm(test_images, desc="Running Inference"):
                filename = os.path.basename(img_path).replace('.jpg', '')
                base_img = cv2.imread(img_path)
                if base_img is None:
                    print(f"Warning: could not read {img_path}")
                    continue
                orig_h, orig_w = base_img.shape[:2]
                
                img_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
                
                img_tensor = img_resized.astype(np.float32) / 255.0
                img_tensor = (img_tensor - mean) / std
                img_tensor = img_tensor.transpose(2, 0, 1)
                
                x = torch.tensor(img_tensor[None, ...], dtype=torch.float32).to(device)
                
                # Predict Synergistic Net Standard
                out_syn_std = syn_model(x)
                p_syn_main_std = F.softmax(out_syn_std['main_output'], dim=1).cpu().numpy()[0]
                p_syn_unet_std = F.softmax(out_syn_std['unet_output'], dim=1).cpu().numpy()[0]
                p_syn_dl_std = F.softmax(out_syn_std['deeplab_output'], dim=1).cpu().numpy()[0]
                
                # Blend calculation
                p_syn_notta = 0.4 * p_syn_main_std + 0.3 * p_syn_unet_std + 0.3 * p_syn_dl_std
                p_syn_notta[0] = p_syn_main_std[0]
                    
                # Predict Meta-Stacked Net
                meta_logits = meta_model(x)
                p_meta = F.softmax(meta_logits, dim=1).cpu().numpy()[0]
                
                # Blend at 480x640 with w_syn = 0.50
                p_blend = 0.50 * p_syn_notta + 0.50 * p_meta
                p_blend[0] = p_syn_notta[0] # keep class 0 unscaled
                
                # Configuration 1: C0=0.95, C3=0.50, C1=0.50, Area=64
                pred_c1 = apply_multiclass_thresholding(p_blend.copy(), {0: 0.95, 3: 0.50, 1: 0.50})
                pred_c1 = neighbor_fill_cleanup(pred_c1.astype(np.uint8), min_area=64)
                
                # Configuration 2: C0=1.00, C3=0.50, C1=0.50, Area=64
                pred_c2 = apply_multiclass_thresholding(p_blend.copy(), {0: 1.00, 3: 0.50, 1: 0.50})
                pred_c2 = neighbor_fill_cleanup(pred_c2.astype(np.uint8), min_area=64)
                
                # Configuration 3: C0=1.00, C3=0.50, C1=0.50, Area=128
                pred_c3 = apply_multiclass_thresholding(p_blend.copy(), {0: 1.00, 3: 0.50, 1: 0.50})
                pred_c3 = neighbor_fill_cleanup(pred_c3.astype(np.uint8), min_area=128)
                
                # Resize final masks to high-res (nearest)
                pred_c1_hr = cv2.resize(pred_c1, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                pred_c2_hr = cv2.resize(pred_c2, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                pred_c3_hr = cv2.resize(pred_c3, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
                classes_c1 = set(np.unique(pred_c1))
                classes_c2 = set(np.unique(pred_c2))
                classes_c3 = set(np.unique(pred_c3))
                
                for class_id in range(10):
                    # Config 1
                    if class_id not in classes_c1:
                        w1.writerow([f"{filename}_{class_id:02d}", ""])
                    else:
                        w1.writerow([f"{filename}_{class_id:02d}", mask2rle((pred_c1_hr == class_id).astype(np.uint8))])
                        
                    # Config 2
                    if class_id not in classes_c2:
                        w2.writerow([f"{filename}_{class_id:02d}", ""])
                    else:
                        w2.writerow([f"{filename}_{class_id:02d}", mask2rle((pred_c2_hr == class_id).astype(np.uint8))])
                        
                    # Config 3
                    if class_id not in classes_c3:
                        w3.writerow([f"{filename}_{class_id:02d}", ""])
                    else:
                        w3.writerow([f"{filename}_{class_id:02d}", mask2rle((pred_c3_hr == class_id).astype(np.uint8))])
                
                del x, out_syn_std, meta_logits, p_syn_notta, p_meta, p_blend, pred_c1, pred_c2, pred_c3, pred_c1_hr, pred_c2_hr, pred_c3_hr
                gc.collect()
                torch.cuda.empty_cache()
            
    del syn_model, meta_model
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\nSUCCESS! Generated No-TTA Candidate Submission CSV files:")
    print(f"  - {out_c1}")
    print(f"  - {out_c2}")
    print(f"  - {out_c3}")


if __name__ == '__main__':
    main()
