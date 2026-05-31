import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"
import glob
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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
    
    # 20% validation split
    image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    _, val_img_paths, _, val_msk_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    print(f"Evaluating on {len(val_img_paths)} validation images at native high-resolution...")
    
    intersection = torch.zeros(10, device=device)
    union = torch.zeros(10, device=device)
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_img_paths))):
            img_bgr = cv2.imread(val_img_paths[idx])
            orig_h, orig_w = img_bgr.shape[:2]
            
            msk = cv2.imread(val_msk_paths[idx])
            msk_rgb = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
            label = rgb_to_mask(msk_rgb, id2color, 10)
            target = torch.tensor(label, dtype=torch.long, device=device)
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            # Prediction
            preds_dict = model(img_tensor)
            logits = preds_dict['main_output']
            
            # TTA logic as in inference_synergistic.py
            img_flipped = torch.flip(img_tensor, dims=[3])
            preds_flipped_dict = model(img_flipped)
            logits_flipped = preds_flipped_dict['main_output']
            
            probs_std = F.softmax(logits, dim=1)
            probs_flip = F.softmax(logits_flipped, dim=1)
            probs_unflip = torch.flip(probs_flip, dims=[3])
            fused_probs = (probs_std + probs_unflip) * 0.5
            fused_probs_np = fused_probs.squeeze(0).cpu().numpy()
            
            # Bilinear resize of probabilities to native size
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(fused_probs_np[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
            pred_labels = torch.tensor(np.argmax(probs_resized, axis=0), device=device)
            
            valid_mask = (target != 255)
            target_safe = torch.where(target == 255, torch.zeros_like(target), target)
            
            for c in range(10):
                pred_c = (pred_labels == c) & valid_mask
                tgt_c = (target_safe == c) & valid_mask
                intersection[c] += torch.sum(pred_c & tgt_c)
                union[c] += torch.sum(pred_c | tgt_c)
                
    dice = (2. * intersection) / (union + intersection + 1e-6)
    dice_np = dice.cpu().numpy()
    
    print("\nClass-wise Dice Scores at Native Resolution:")
    for c in range(10):
        print(f"Class {c}: {dice_np[c]:.4f}")
    print(f"Mean Dice (All 10 classes): {np.mean(dice_np):.4f}")
    print(f"Mean Dice (Classes 1-9): {np.mean(dice_np[1:]):.4f}")

if __name__ == '__main__':
    main()
