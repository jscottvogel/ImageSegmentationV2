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

# Setup validation split
image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))

_, val_image_paths, _, val_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        img = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.cvtColor(cv2.imread(self.msk_paths[idx]), cv2.COLOR_BGR2RGB)
        
        img_resized = cv2.resize(img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
        img_tensor = torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        
        label = rgb_to_mask(msk, id2color, 10)
        target = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, target

val_dataset = EvalDataset(val_image_paths, val_mask_paths)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

def normalize_imagenet(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (img_tensor - mean) / std

def eval_checkpoint(path):
    print(f"Evaluating {path}...")
    model = CustomDeepLabV3Plus(num_classes=10).to(device)
    model = load_weights_custom(model, path, device)
    
    intersection = torch.zeros(10, device=device)
    union = torch.zeros(10, device=device)
    
    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc="Eval"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            norm_imgs = normalize_imagenet(imgs)
            
            out = model(norm_imgs)
            preds = torch.argmax(out['main_output'], dim=1)
            
            valid_mask = (targets != 255)
            targets_safe = torch.where(targets == 255, torch.zeros_like(targets), targets)
            for c in range(10):
                intersection[c] += torch.sum((preds == c) & valid_mask & (targets_safe == c))
                union[c] += torch.sum(((preds == c) | (targets_safe == c)) & valid_mask)
                
    dice = (2. * intersection) / (union + intersection + 1e-6)
    print(f"Mean Dice (Class 1-9): {np.mean(dice[1:].cpu().numpy()):.6f}")
    return np.mean(dice[1:].cpu().numpy())

if __name__ == '__main__':
    ckpts = [
        "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt",
        "model_checkpoint/FloodNet_PyTorch/final_swa_smoothed_weights.pt",
        "model_checkpoint/FloodNet_PyTorch_V2S_Backup/best_deeplab_weights.pt",
        "model_checkpoint/FloodNet_PyTorch_V2S_Backup/final_swa_smoothed_weights.pt",
    ]
    for ckpt in ckpts:
        if os.path.exists(ckpt):
            eval_checkpoint(ckpt)
        else:
            print(f"Path does not exist: {ckpt}")
