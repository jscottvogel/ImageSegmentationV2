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

from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig, id2color, rgb_to_mask
from unet_version import StandardUNet
from fcn_version import ResNet50FCN

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load models
    deeplab = CustomDeepLabV3Plus(num_classes=10).to(device)
    deeplab = load_weights_custom(deeplab, "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt", device)
    
    unet = StandardUNet(num_classes=10).to(device)
    unet = load_weights_custom(unet, "model_checkpoint/FloodNet_UNet/best_unet_weights.pt", device)
    
    fcn = ResNet50FCN(num_classes=10).to(device)
    fcn = load_weights_custom(fcn, "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt", device)
    
    meta_layer = None
    meta_weights_path = "model_checkpoint/FloodNet_Meta/meta_layer_weights.pt"
    if os.path.exists(meta_weights_path):
        meta_state = torch.load(meta_weights_path, map_location=device, weights_only=True)
        meta_layer = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=1).to(device)
        meta_layer.load_state_dict(meta_state)
        meta_layer.eval()
        
    # 20% validation split
    image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    _, val_img_paths, _, val_msk_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    print(f"Evaluating Meta Ensemble on {len(val_img_paths)} validation images at native high-resolution...")
    
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
            
            # Predict standard
            norm_img = img_tensor
            out_dl_std = F.softmax(deeplab(norm_img)['main_output'], dim=1)
            
            out_unet_std = unet(norm_img)
            if isinstance(out_unet_std, dict):
                out_unet_std = out_unet_std['main_output']
            out_unet_std = F.softmax(out_unet_std, dim=1)
            
            out_fcn_std = fcn(norm_img)
            if isinstance(out_fcn_std, dict):
                out_fcn_std = out_fcn_std['main_output']
            out_fcn_std = F.softmax(out_fcn_std, dim=1)
            
            # Predict TTA flip
            img_flipped = torch.flip(img_tensor, dims=[3])
            out_dl_flip = F.softmax(deeplab(img_flipped)['main_output'], dim=1)
            out_dl_unflip = torch.flip(out_dl_flip, dims=[3])
            
            out_unet_flip = unet(img_flipped)
            if isinstance(out_unet_flip, dict):
                out_unet_flip = out_unet_flip['main_output']
            out_unet_flip = F.softmax(out_unet_flip, dim=1)
            out_unet_unflip = torch.flip(out_unet_flip, dims=[3])
            
            out_fcn_flip = fcn(img_flipped)
            if isinstance(out_fcn_flip, dict):
                out_fcn_flip = out_fcn_flip['main_output']
            out_fcn_flip = F.softmax(out_fcn_flip, dim=1)
            out_fcn_unflip = torch.flip(out_fcn_flip, dims=[3])
            
            fused_probs_dl = (out_dl_std + out_dl_unflip) * 0.5
            fused_probs_unet = (out_unet_std + out_unet_unflip) * 0.5
            fused_probs_fcn = (out_fcn_std + out_fcn_unflip) * 0.5
            
            # Stack and apply meta layer
            stacked_logits = torch.cat([fused_probs_unet, fused_probs_fcn, fused_probs_dl], dim=1)
            meta_logits = meta_layer(stacked_logits)
            p_meta = F.softmax(meta_logits, dim=1).squeeze(0).cpu().numpy()
            
            # Bilinear resize of probabilities to native size
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(p_meta[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
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
    
    print("\nMeta Ensemble Class-wise Dice Scores at Native Resolution:")
    for c in range(10):
        print(f"Class {c}: {dice_np[c]:.4f}")
    print(f"Mean Dice (All 10 classes): {np.mean(dice_np):.4f}")
    print(f"Mean Dice (Classes 1-9): {np.mean(dice_np[1:]):.4f}")

if __name__ == '__main__':
    main()
