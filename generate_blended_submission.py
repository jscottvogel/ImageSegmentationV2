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

from synergistic_model import FloodNetSynergisticNet
from competitive_model import FloodNetCompetitiveModel
from optimized_pytorch_version import DatasetConfig
from scratch.eval_advanced_tta import neighbor_fill_cleanup

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

def main():
    print("Initializing Optimal Blended Ensemble Inference Pipeline in FP32...")
    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    SUBMISSION_PATH = "blended_ensemble_submission.csv"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the Synergistic model
    print("Loading Synergistic Model...")
    syn_model = FloodNetSynergisticNet(num_classes=10).to(device)
    syn_weights = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    if os.path.exists(syn_weights):
        state = torch.load(syn_weights, map_location=device, weights_only=True)
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
        syn_model.load_state_dict(state, strict=False)
    syn_model.eval()
    
    # 2. Load the Meta-Stacked model
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
    
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    
    if "DRY_RUN" in os.environ:
        print("[DRY RUN] Limiting inference to first 5 images.")
        test_images = test_images[:5]
        
    if len(test_images) == 0:
        print("CRITICAL ERROR: No test images found!")
        return
        
    submission_t30_data = []
    submission_t95_data = []
    submission_t99_data = []
    
    print(f"Running Blended Inference on {len(test_images)} test images...")
    with torch.no_grad():
        for idx, img_path in enumerate(tqdm(test_images)):
            filename = os.path.basename(img_path).replace('.jpg', '')
            base_img = cv2.imread(img_path)
            orig_h, orig_w = base_img.shape[:2]
            
            # Preprocessing matching dataset
            img_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            # --- Predict Synergistic Net with TTA (horizontal flip) ---
            img_flipped = torch.flip(img_tensor, dims=[3])
            out_syn_std = syn_model(img_tensor)
            out_syn_flip = syn_model(img_flipped)
            
            p_syn_main_std = F.softmax(out_syn_std['main_output'], dim=1)
            p_syn_main_flip = F.softmax(out_syn_flip['main_output'], dim=1)
            p_syn_main = (p_syn_main_std + torch.flip(p_syn_main_flip, dims=[3])) * 0.5
            p_syn_main_np = p_syn_main.cpu().squeeze(0).numpy()
            
            p_syn_unet_std = F.softmax(out_syn_std['unet_output'], dim=1)
            p_syn_unet_flip = F.softmax(out_syn_flip['unet_output'], dim=1)
            p_syn_unet = (p_syn_unet_std + torch.flip(p_syn_unet_flip, dims=[3])) * 0.5
            p_syn_unet_np = p_syn_unet.cpu().squeeze(0).numpy()
            
            p_syn_dl_std = F.softmax(out_syn_std['deeplab_output'], dim=1)
            p_syn_dl_flip = F.softmax(out_syn_flip['deeplab_output'], dim=1)
            p_syn_dl = (p_syn_dl_std + torch.flip(p_syn_dl_flip, dims=[3])) * 0.5
            p_syn_dl_np = p_syn_dl.cpu().squeeze(0).numpy()
            
            p_syn = 0.6 * p_syn_main_np + 0.2 * p_syn_unet_np + 0.2 * p_syn_dl_np
            p_syn[0] = p_syn_main_np[0] # keep class 0 unscaled
            
            # --- Predict Meta-Stacked Net with TTA ---
            meta_logits = meta_model(img_tensor)
            p_meta = F.softmax(meta_logits, dim=1).squeeze(0).cpu().numpy()
            
            # --- Blend Predictions ---
            p_blend = 0.7 * p_syn + 0.3 * p_meta
            
            # Resize probability maps to original size
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(p_blend[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                
            pred_base = np.argmax(probs_resized, axis=0)
            class0_mask = (pred_base == 0)
            fallback_labels = np.argmax(probs_resized[1:], axis=0) + 1
            
            # --- T=0.30 ---
            pred_t30 = pred_base.copy()
            low_conf_t30 = class0_mask & (probs_resized[0] < 0.30)
            if np.any(low_conf_t30):
                pred_t30[low_conf_t30] = fallback_labels[low_conf_t30]
            pred_t30 = neighbor_fill_cleanup(pred_t30.astype(np.uint8), min_area=25)
            
            # --- T=0.95 ---
            pred_t95 = pred_base.copy()
            low_conf_t95 = class0_mask & (probs_resized[0] < 0.95)
            if np.any(low_conf_t95):
                pred_t95[low_conf_t95] = fallback_labels[low_conf_t95]
            pred_t95 = neighbor_fill_cleanup(pred_t95.astype(np.uint8), min_area=25)
            
            # --- T=0.99 ---
            pred_t99 = pred_base.copy()
            low_conf_t99 = class0_mask & (probs_resized[0] < 0.99)
            if np.any(low_conf_t99):
                pred_t99[low_conf_t99] = fallback_labels[low_conf_t99]
            pred_t99 = neighbor_fill_cleanup(pred_t99.astype(np.uint8), min_area=25)
            
            # Class RLE encoding
            for class_id in range(10):
                submission_t30_data.append([f"{filename}_{class_id:02d}", mask2rle((pred_t30 == class_id).astype(np.uint8))])
                submission_t95_data.append([f"{filename}_{class_id:02d}", mask2rle((pred_t95 == class_id).astype(np.uint8))])
                submission_t99_data.append([f"{filename}_{class_id:02d}", mask2rle((pred_t99 == class_id).astype(np.uint8))])
                
            # Clear memory per image
            del img_tensor, img_flipped, out_syn_std, out_syn_flip, p_syn_main_std, p_syn_main_flip
            del p_syn_unet_std, p_syn_unet_flip, p_syn_dl_std, p_syn_dl_flip, meta_logits
            del p_syn, p_meta, p_blend, probs_resized, pred_base, pred_t30, pred_t95, pred_t99
            gc.collect()
            torch.cuda.empty_cache()
            
    out_t30 = "blended_ensemble_t30_submission.csv"
    out_t95 = "blended_ensemble_t95_submission.csv"
    out_t99 = "blended_ensemble_t99_submission.csv"
    
    pd.DataFrame(submission_t30_data, columns=["IMG_ID", "EncodedString"]).to_csv(out_t30, index=False)
    pd.DataFrame(submission_t95_data, columns=["IMG_ID", "EncodedString"]).to_csv(out_t95, index=False)
    pd.DataFrame(submission_t99_data, columns=["IMG_ID", "EncodedString"]).to_csv(out_t99, index=False)
    print(f"\nSUCCESS! Blended Ensemble Submissions Generated at:\n  - {out_t30}\n  - {out_t95}\n  - {out_t99}")
    print(f"\nSUCCESS! Blended Ensemble Submission CSV Generated at: {SUBMISSION_PATH}")

if __name__ == '__main__':
    main()
