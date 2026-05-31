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

# Setup validation split
image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))

_, val_image_paths, _, val_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

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
        
        orig_h, orig_w = img.shape[:2]
        
        # We need to resize to (480, 640) for the network
        img_resized = cv2.resize(img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
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

def eval_model(model, model_type='dl', mode_fusion=False, edge_sharpening=False):
    intersection = torch.zeros(10, device=device)
    union = torch.zeros(10, device=device)
    
    with torch.no_grad():
        for imgs, targets, orig_hs, orig_ws in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            norm_imgs = normalize_imagenet(imgs)
            
            # Predict
            out = model(norm_imgs)
            if model_type == 'dl':
                if not mode_fusion:
                    probs = F.softmax(out['main_output'], dim=1)
                else:
                    main_p = F.softmax(out['main_output'], dim=1)
                    side_p = F.softmax(out['side_output'], dim=1)
                    mid_p = F.softmax(out['mid_output'], dim=1)
                    high_p = F.softmax(out['high_output'], dim=1)
                    low_p = F.softmax(out['low_output'], dim=1)
                    mid_high_p = F.softmax(out['mid_high_output'], dim=1)
                    
                    probs = 0.4 * main_p + 0.2 * side_p + 0.2 * mid_p + 0.2 * high_p + 0.2 * low_p + 0.2 * mid_high_p
                    
                    if edge_sharpening:
                        edge_p = torch.sigmoid(out['edge_output'])
                        probs *= (1.0 + edge_p)
                    probs /= (probs.sum(dim=1, keepdim=True) + 1e-6)
            else:
                if isinstance(out, dict):
                    out = out['main_output']
                probs = F.softmax(out, dim=1)
                
            probs_resized = F.interpolate(probs, size=(480, 640), mode='bilinear', align_corners=False)
            pred_labels = torch.argmax(probs_resized, dim=1)
            
            valid_mask = (targets != 255)
            targets_safe = torch.where(targets == 255, torch.zeros_like(targets), targets)
            
            for c in range(10):
                pred_c = (pred_labels == c) & valid_mask
                tgt_c = (targets_safe == c) & valid_mask
                intersection[c] += torch.sum(pred_c & tgt_c)
                union[c] += torch.sum(pred_c | tgt_c)
                
    dice = (2. * intersection) / (union + intersection + 1e-6)
    return dice.cpu().numpy()

if __name__ == '__main__':
    print("Loading DeepLab models...")
    dl_best = CustomDeepLabV3Plus(num_classes=10).to(device)
    dl_best = load_weights_custom(dl_best, "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt", device)
    
    dl_swa = CustomDeepLabV3Plus(num_classes=10).to(device)
    dl_swa = load_weights_custom(dl_swa, "model_checkpoint/FloodNet_PyTorch/final_swa_smoothed_weights.pt", device)
    
    print("\n--- DeepLab Baseline Evaluations ---")
    
    # 1. DeepLab Best (no fusion)
    dice = eval_model(dl_best, 'dl', mode_fusion=False, edge_sharpening=False)
    print(f"DeepLab Best (no fusion) | Class 1-9 Avg Dice: {np.mean(dice[1:]):.4f}")
    
    # 2. DeepLab Best (with fusion, no edge sharpening)
    dice = eval_model(dl_best, 'dl', mode_fusion=True, edge_sharpening=False)
    print(f"DeepLab Best (with fusion, no edge) | Class 1-9 Avg Dice: {np.mean(dice[1:]):.4f}")
    
    # 3. DeepLab Best (with fusion + edge sharpening)
    dice = eval_model(dl_best, 'dl', mode_fusion=True, edge_sharpening=True)
    print(f"DeepLab Best (with fusion + edge) | Class 1-9 Avg Dice: {np.mean(dice[1:]):.4f}")
    
    # 4. DeepLab SWA (with fusion + edge sharpening)
    dice = eval_model(dl_swa, 'dl', mode_fusion=True, edge_sharpening=True)
    print(f"DeepLab SWA (with fusion + edge) | Class 1-9 Avg Dice: {np.mean(dice[1:]):.4f}")
    
    print("Loading UNet and FCN models...")
    unet_best = StandardUNet(num_classes=10).to(device)
    unet_best = load_weights_custom(unet_best, "model_checkpoint/FloodNet_UNet/best_unet_weights.pt", device)
    
    unet_swa = StandardUNet(num_classes=10).to(device)
    unet_swa = load_weights_custom(unet_swa, "model_checkpoint/FloodNet_UNet/final_swa_smoothed_unet.pt", device)
    
    fcn_best = ResNet50FCN(num_classes=10).to(device)
    fcn_best = load_weights_custom(fcn_best, "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt", device)
    
    fcn_swa = ResNet50FCN(num_classes=10).to(device)
    fcn_swa = load_weights_custom(fcn_swa, "model_checkpoint/FloodNet_FCN/final_swa_smoothed_fcn.pt", device)
    
    print("\n--- UNet & FCN evaluations ---")
    dice = eval_model(unet_best, 'unet')
    print(f"UNet Best | Class 1-9 Avg Dice: {np.mean(dice[1:]):.4f}")
    
    dice = eval_model(unet_swa, 'unet')
    print(f"UNet SWA | Class 1-9 Avg Dice: {np.mean(dice[1:]):.4f}")
    
    dice = eval_model(fcn_best, 'fcn')
    print(f"FCN Best | Class 1-9 Avg Dice: {np.mean(dice[1:]):.4f}")
    
    dice = eval_model(fcn_swa, 'fcn')
    print(f"FCN SWA | Class 1-9 Avg Dice: {np.mean(dice[1:]):.4f}")
