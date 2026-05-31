import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from synergistic_model import FloodNetSynergisticNet
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FloodNetSynergisticNet(num_classes=10).to(device)
    
    weights_path = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    state = torch.load(weights_path, map_location=device, weights_only=True)
    state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
    model.load_state_dict(state)
    model.eval()
    
    image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    _, val_img_paths, _, val_msk_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    val_img_paths = val_img_paths[:100]
    val_msk_paths = val_msk_paths[:100]
    
    gt_class0_count = 0
    gt_class0_pixels = 0
    
    pred_class0_count = 0
    pred_class0_pixels = 0
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_img_paths))):
            msk = cv2.imread(val_msk_paths[idx])
            msk_rgb = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
            label = rgb_to_mask(msk_rgb, id2color, 10)
            
            c0_gt_pixels = np.sum(label == 0)
            gt_class0_pixels += c0_gt_pixels
            if c0_gt_pixels > 0:
                gt_class0_count += 1
                
            img_bgr = cv2.imread(val_img_paths[idx])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            preds_dict = model(img_tensor)
            logits = preds_dict['main_output']
            
            # TTA horizontal flip
            img_flipped = torch.flip(img_tensor, dims=[3])
            preds_flipped_dict = model(img_flipped)
            logits_flipped = preds_flipped_dict['main_output']
            
            probs_std = torch.softmax(logits, dim=1)
            probs_flip = torch.softmax(logits_flipped, dim=1)
            probs_unflip = torch.flip(probs_flip, dims=[3])
            fused_probs = (probs_std + probs_unflip) * 0.5
            fused_probs_np = fused_probs.squeeze(0).cpu().numpy()
            
            probs_resized = np.zeros((10, msk.shape[0], msk.shape[1]), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(fused_probs_np[c], (msk.shape[1], msk.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            pred_labels = np.argmax(probs_resized, axis=0)
            c0_pred_pixels = np.sum(pred_labels == 0)
            pred_class0_pixels += c0_pred_pixels
            if c0_pred_pixels > 0:
                pred_class0_count += 1
                
    print(f"\nValidation Ground Truth (100 images):")
    print(f"  Images with Class 0: {gt_class0_count} | Total Class 0 pixels: {gt_class0_pixels}")
    print(f"Validation Predictions (Threshold 0.00, 100 images):")
    print(f"  Images with Class 0: {pred_class0_count} | Total Class 0 pixels: {pred_class0_pixels}")

if __name__ == '__main__':
    main()
