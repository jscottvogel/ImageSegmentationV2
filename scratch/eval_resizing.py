import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2

from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig, id2color, rgb_to_mask
from unet_version import StandardUNet
from fcn_version import ResNet50FCN

image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))

_, val_image_paths, _, val_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)
val_image_paths = val_image_paths[:50]
val_mask_paths = val_mask_paths[:50]

device = torch.device('cpu')

def load_weights_custom(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if 'n_averaged' in state:
        del state['n_averaged']
    clean_state = {}
    for k, v in state.items():
        k = k.replace('module.', '').replace('_orig_mod.', '')
        clean_state[k] = v
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, msk_paths):
        self.img_paths = img_paths
        self.msk_paths = msk_paths
        
    def __len__(self):
        return len(self.img_paths)
        
    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx]) # Keep original BGR or RGB? opencv loads BGR. But we need RGB for model!
        # Wait, the model expects RGB!
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = cv2.cvtColor(cv2.imread(self.msk_paths[idx]), cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = img.shape[:2]
        
        img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
        img_tensor = torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        
        label = rgb_to_mask(msk, id2color, 10)
        target = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, target, orig_h, orig_w

val_dataset = EvalDataset(val_image_paths, val_mask_paths)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

def normalize_imagenet(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (img_tensor - mean) / std

if __name__ == '__main__':
    print("Loading UNet Best Model...")
    unet = StandardUNet(num_classes=10).to(device)
    unet = load_weights_custom(unet, "model_checkpoint/FloodNet_UNet/best_unet_weights.pt", device)
    
    intersection_bilinear = torch.zeros(10, device=device)
    union_bilinear = torch.zeros(10, device=device)
    
    intersection_nearest = torch.zeros(10, device=device)
    union_nearest = torch.zeros(10, device=device)

    with torch.no_grad():
        for imgs, targets, orig_hs, orig_ws in tqdm(val_loader, desc="Evaluation"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            norm_imgs = normalize_imagenet(imgs)
            
            # Predict UNet
            out_unet = unet(norm_imgs)
            if isinstance(out_unet, dict):
                out_unet = out_unet['main_output']
            p_unet = F.softmax(out_unet, dim=1) # (B, 10, 480, 640)
            
            # We must evaluate at original resolution to compare resizing methods!
            # Since different batch elements can have different original shapes, let's process one by one
            for i in range(imgs.size(0)):
                oh, ow = orig_hs[i].item(), orig_ws[i].item()
                p_img = p_unet[i].cpu().numpy() # (10, 480, 640)
                tgt_img = targets[i].cpu().numpy() # (oh, ow)
                
                # Method A: Resize probabilities (bilinear) -> Argmax
                p_resized = np.zeros((10, oh, ow), dtype=np.float32)
                for c in range(10):
                    p_resized[c] = cv2.resize(p_img[c], (ow, oh), interpolation=cv2.INTER_LINEAR)
                pred_bilinear = np.argmax(p_resized, axis=0)
                
                # Method B: Argmax -> Resize mask (nearest)
                pred_low = np.argmax(p_img, axis=0) # (480, 640)
                pred_nearest = cv2.resize(pred_low.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST)
                
                valid_mask = (tgt_img != 255)
                tgt_safe = np.where(tgt_img == 255, 0, tgt_img)
                
                for c in range(10):
                    intersection_bilinear[c] += np.sum((pred_bilinear == c) & valid_mask & (tgt_safe == c))
                    union_bilinear[c] += np.sum(((pred_bilinear == c) | (tgt_safe == c)) & valid_mask)
                    
                    intersection_nearest[c] += np.sum((pred_nearest == c) & valid_mask & (tgt_safe == c))
                    union_nearest[c] += np.sum(((pred_nearest == c) | (tgt_safe == c)) & valid_mask)
                    
    dice_bilinear = (2. * intersection_bilinear) / (union_bilinear + intersection_bilinear + 1e-6)
    dice_nearest = (2. * intersection_nearest) / (union_nearest + intersection_nearest + 1e-6)
    
    print("\n--- Resizing Comparison (Class 1-9 Avg Dice) ---")
    print(f"Method A (Bilinear on probs): {np.mean(dice_bilinear[1:].cpu().numpy()):.6f}")
    print(f"Method B (Nearest on mask):  {np.mean(dice_nearest[1:].cpu().numpy()):.6f}")
