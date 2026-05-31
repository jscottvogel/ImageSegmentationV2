import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"
import glob
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import gc
from tqdm import tqdm

from competitive_model import FloodNetCompetitiveModel
from optimized_pytorch_version import DatasetConfig

# 1-based RLE encoder (Kaggle compliant)
def mask2rle(img: np.ndarray) -> str:
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]
    runs[1::2] -= runs[::2]
    runs[::2] += 1
    if len(runs) == 0:
        return ""
    return ' '.join(str(x) for x in runs)

def normalize_imagenet(img_tensor, device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (img_tensor - mean) / std

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Final Submission Pipeline with competitive model on: {device}")
    
    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    
    if len(test_images) == 0:
        print("CRITICAL ERROR: No test images found!")
        return
        
    print(f"Found {len(test_images)} test images.")
    
    # 1. Load competitive model and weights
    model = FloodNetCompetitiveModel(num_classes=10).to(device)
    unet_path = "model_checkpoint/FloodNet_UNet/best_unet_weights.pt"
    fcn_path = "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt"
    deeplab_path = "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt"
    meta_path = "model_checkpoint/FloodNet_Meta/meta_layer_weights.pt"
    
    model.load_checkpoints(
        unet_path=unet_path,
        fcn_path=fcn_path,
        deeplab_path=deeplab_path,
        meta_path=meta_path,
        device=device
    )
    
    model.eval()
    
    # 2. Submission container
    submission = []
    
    # 3. Processing loop
    for img_path in tqdm(test_images, desc="Generating final predictions"):
        filename = os.path.basename(img_path).replace('.jpg', '')
        
        # Load and preprocess
        img_bgr = cv2.imread(img_path)
        orig_h, orig_w = img_bgr.shape[:2]
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
        
        img_tensor = torch.tensor(resized_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        norm_img = normalize_imagenet(img_tensor, device)
        
        # Forward pass with native TTA
        logits = model(norm_img) # shape: [1, 10, H, W]
        
        # Move logits to CPU immediately (10 x 480 x 640 x 4 bytes = 12 MB)
        logits_cpu = logits.squeeze(0).cpu().numpy()
        
        # Free GPU logits
        del logits
        torch.cuda.empty_cache()
        
        # Incremental channel-wise resizing and argmax on CPU to avoid massive VRAM allocation
        max_val = np.full((orig_h, orig_w), -1e9, dtype=np.float32)
        labels = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        for c in range(10):
            resized_channel = cv2.resize(logits_cpu[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            mask = resized_channel > max_val
            labels[mask] = c
            max_val[mask] = resized_channel[mask]
            del resized_channel, mask
            
        # Clean up temporary arrays
        del max_val, logits_cpu
        
        # Optimize RLE encoding by checking which classes are present in the image
        present_classes = set(np.unique(labels))
        
        # Save RLE for each class (0 to 9)
        for class_id in range(10):
            if class_id not in present_classes:
                encoded = ""
            else:
                encoded = mask2rle((labels == class_id).astype(np.uint8))
            submission.append([f"{filename}_{class_id:02d}", encoded])
            
        # Clean memory
        del img_tensor, norm_img, labels, present_classes
        gc.collect()
        torch.cuda.empty_cache()
        
    # 4. Save submission
    out_csv = "final_competitive_submission.csv"
    print(f"\nSaving final submission to {out_csv}...")
    pd.DataFrame(submission, columns=["IMG_ID", "EncodedString"]).to_csv(out_csv, index=False)
    print("Done!")

if __name__ == '__main__':
    main()
