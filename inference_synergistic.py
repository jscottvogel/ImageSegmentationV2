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
import matplotlib.pyplot as plt
import gc

from synergistic_model import FloodNetSynergisticNet
from optimized_pytorch_version import DatasetConfig

id2color = {
    0: [0, 0, 0], 1: [255, 0, 0], 2: [200, 90, 90], 3: [128, 128, 0], 4: [155, 155, 155],
    5: [0, 255, 255], 6: [55, 0, 255], 7: [255, 0, 255], 8: [245, 245, 0], 9: [0, 255, 0],
}

def decode_segmap(image, nc=10):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(nc):
        idx = image == l
        r[idx] = id2color[l][0]
        g[idx] = id2color[l][1]
        b[idx] = id2color[l][2]
    return np.stack([r, g, b], axis=2)

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

def safe_morphological_cleanup(pred_labels: np.ndarray, min_area=50) -> np.ndarray:
    if min_area <= 0:
        return pred_labels
    clean_labels = pred_labels.copy()
    for class_id in np.unique(pred_labels):
        if class_id == 0: continue
        class_mask = (pred_labels == class_id).astype(np.uint8)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for i in range(1, num_components):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                clean_labels[labels == i] = 0
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

def run_synergistic_inference():
    print("Initialize Synergistic Net Inference Pipeline...")
    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    SUBMISSION_PATH = "synergistic_submission.csv"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the Synergistic model
    model = FloodNetSynergisticNet(num_classes=10).to(device)
    
    checkpoint_dir = "model_checkpoint/FloodNet_Synergistic"
    swa_weights = os.path.join(checkpoint_dir, "final_swa_smoothed_synergistic.pt")
    best_weights = os.path.join(checkpoint_dir, "best_synergistic_weights.pt")
    
    weights_path = None
    if os.path.exists(best_weights):
        weights_path = best_weights
    elif os.path.exists(swa_weights):
        weights_path = swa_weights
    else:
        print("[WARNING] No trained synergistic weights found! Reverting to check using random weights.")
        
    if weights_path:
        print(f"Loading weights from: {weights_path}")
        state = torch.load(weights_path, map_location=device, weights_only=True)
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
        model.load_state_dict(state)
    
    model.eval()
    
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    if "DRY_RUN" in os.environ:
        print("[DRY RUN] Limiting inference to first 5 images.")
        test_images = test_images[:5]
    if len(test_images) == 0:
        print("CRITICAL ERROR: No test images found!")
        return
        
    submission_data = []
    os.makedirs("visualizations_synergistic", exist_ok=True)
    
    print(f"Running Inference on {len(test_images)} targets...")
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
        
        with torch.no_grad():
            fused_probs = multiscale_tta_inference(model, img_tensor)
            
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(fused_probs[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            pred_labels = np.argmax(probs_resized, axis=0).astype(np.uint8)
            pred_labels = safe_morphological_cleanup(pred_labels, min_area=50)
            
        # Class RLE encoding
        present_classes = set(np.unique(pred_labels))
        for class_id in range(10):
            if class_id not in present_classes:
                encoded_string = ""
            else:
                binary_mask = (pred_labels == class_id).astype(np.uint8)
                encoded_string = mask2rle(binary_mask)
            submission_data.append([f"{filename}_{class_id:02d}", encoded_string])
            
        # Visualize check periodically
        if idx % 20 == 0:
            fig = plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(img_rgb)
            plt.title(f"Original: {filename}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(decode_segmap(pred_labels))
            plt.title("Synergistic Prediction")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"visualizations_synergistic/check_{filename}.jpg", bbox_inches='tight')
            plt.close(fig)
            
        # Release memory
        del img_tensor, fused_probs, probs_resized, pred_labels, present_classes
        gc.collect()
        torch.cuda.empty_cache()
        
    submission_df = pd.DataFrame(submission_data, columns=["IMG_ID", "EncodedString"])
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSUCCESS! Synergistic Model Submission CSV Generated AT: {SUBMISSION_PATH}")

if __name__ == '__main__':
    run_synergistic_inference()
