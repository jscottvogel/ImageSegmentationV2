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

from unet_version import StandardUNet
from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig, id2color, rgb_to_mask
from fcn_version import ResNet50FCN
from synergistic_model import FloodNetSynergisticNet

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

def row_wise_dice(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    sum_pred = np.sum(pred_mask)
    sum_tgt = np.sum(target_mask)
    if sum_pred == 0 and sum_tgt == 0:
        return 1.0
    if sum_pred == 0 or sum_tgt == 0:
        return 0.0
    intersection = np.sum(pred_mask & target_mask)
    return (2.0 * intersection) / (sum_pred + sum_tgt)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Synergistic Net
    syn_model = FloodNetSynergisticNet(num_classes=10).to(device)
    syn_model = load_weights_custom(syn_model, "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt", device)
    
    # 2. Load Meta Ensemble components
    deeplab = CustomDeepLabV3Plus(num_classes=10).to(device)
    deeplab = load_weights_custom(deeplab, "model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt", device)
    
    unet = StandardUNet(num_classes=10).to(device)
    unet = load_weights_custom(unet, "model_checkpoint/FloodNet_UNet/best_unet_weights.pt", device)
    
    fcn = ResNet50FCN(num_classes=10).to(device)
    fcn = load_weights_custom(fcn, "model_checkpoint/FloodNet_FCN/best_fcn_weights.pt", device)
    
    meta_layer = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=1).to(device)
    meta_state = torch.load("model_checkpoint/FloodNet_Meta/meta_layer_weights.pt", map_location=device, weights_only=True)
    meta_layer.load_state_dict(meta_state)
    meta_layer.eval()
    
    # Validation split
    image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    _, val_img_paths, _, val_msk_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    # Evaluate on first 50 validation images
    val_img_paths = val_img_paths[:50]
    val_msk_paths = val_msk_paths[:50]
    
    print(f"Precomputing predictions for {len(val_img_paths)} validation images...")
    
    syn_probs_list = []
    meta_probs_list = []
    targets_list = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_img_paths))):
            img_bgr = cv2.imread(val_img_paths[idx])
            orig_h, orig_w = img_bgr.shape[:2]
            
            msk = cv2.imread(val_msk_paths[idx])
            msk_rgb = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
            label = rgb_to_mask(msk_rgb, id2color, 10)
            targets_list.append(label)
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            # Predict Synergistic Net with TTA
            img_flipped = torch.flip(img_tensor, dims=[3])
            syn_out_std = F.softmax(syn_model(img_tensor)['main_output'], dim=1)
            syn_out_flip = F.softmax(syn_model(img_flipped)['main_output'], dim=1)
            syn_out_unflip = torch.flip(syn_out_flip, dims=[3])
            syn_fused = (syn_out_std + syn_out_unflip) * 0.5
            syn_fused_np = syn_fused.squeeze(0).cpu().numpy()
            
            syn_probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                syn_probs_resized[c] = cv2.resize(syn_fused_np[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            syn_probs_list.append(syn_probs_resized)
            
            # Predict Meta Ensemble with TTA
            dl_std = F.softmax(deeplab(img_tensor)['main_output'], dim=1)
            unet_std = unet(img_tensor)
            if isinstance(unet_std, dict): unet_std = unet_std['main_output']
            unet_std = F.softmax(unet_std, dim=1)
            fcn_std = fcn(img_tensor)
            if isinstance(fcn_std, dict): fcn_std = fcn_std['main_output']
            fcn_std = F.softmax(fcn_std, dim=1)
            
            dl_flip = F.softmax(deeplab(img_flipped)['main_output'], dim=1)
            dl_unflip = torch.flip(dl_flip, dims=[3])
            unet_flip = unet(img_flipped)
            if isinstance(unet_flip, dict): unet_flip = unet_flip['main_output']
            unet_flip = F.softmax(unet_flip, dim=1)
            unet_unflip = torch.flip(unet_flip, dims=[3])
            fcn_flip = fcn(img_flipped)
            if isinstance(fcn_flip, dict): fcn_flip = fcn_flip['main_output']
            fcn_flip = F.softmax(fcn_flip, dim=1)
            fcn_unflip = torch.flip(fcn_flip, dims=[3])
            
            dl_fused = (dl_std + dl_unflip) * 0.5
            unet_fused = (unet_std + unet_unflip) * 0.5
            fcn_fused = (fcn_std + fcn_unflip) * 0.5
            
            stacked_logits = torch.cat([unet_fused, fcn_fused, dl_fused], dim=1)
            meta_logits = meta_layer(stacked_logits)
            meta_probs = F.softmax(meta_logits, dim=1).squeeze(0).cpu().numpy()
            
            meta_probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                meta_probs_resized[c] = cv2.resize(meta_probs[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            meta_probs_list.append(meta_probs_resized)
            
    # Grid search blending weights and Class 0 thresholds
    blend_weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    
    results = []
    print("\nRunning grid search...")
    for w_syn in blend_weights:
        w_meta = 1.0 - w_syn
        for thresh in thresholds:
            row_dices = {c: [] for c in range(10)}
            
            for syn_p, meta_p, target in zip(syn_probs_list, meta_probs_list, targets_list):
                # Blend probabilities
                probs = w_syn * syn_p + w_meta * meta_p
                
                # Apply thresholding
                if thresh == 0.0:
                    pred_labels = np.argmax(probs, axis=0)
                elif thresh == 1.0:
                    pred_labels = np.argmax(probs[1:], axis=0) + 1
                else:
                    pred_labels = np.argmax(probs, axis=0)
                    class0_mask = (pred_labels == 0)
                    low_conf_mask = class0_mask & (probs[0] < thresh)
                    if np.any(low_conf_mask):
                        fallback_labels = np.argmax(probs[1:], axis=0) + 1
                        pred_labels[low_conf_mask] = fallback_labels[low_conf_mask]
                        
                valid_mask = (target != 255)
                for c in range(10):
                    pred_c = (pred_labels == c) & valid_mask
                    tgt_c = (target == c) & valid_mask
                    row_dices[c].append(row_wise_dice(pred_c, tgt_c))
                    
            class_means = {c: np.mean(row_dices[c]) for c in range(10)}
            overall_mean = np.mean([class_means[c] for c in range(10)])
            class_1_9_mean = np.mean([class_means[c] for c in range(1, 10)])
            
            results.append({
                'w_syn': w_syn,
                'w_meta': w_meta,
                'thresh': thresh,
                'mean_dice': overall_mean,
                'class_1_9': class_1_9_mean,
                'class_0': class_means[0]
            })
            
    # Sort results
    results = sorted(results, key=lambda x: x['mean_dice'], reverse=True)
    
    print("\n--- TOP 20 GRID SEARCH CONFIGURATIONS ---")
    for idx, r in enumerate(results[:20]):
        print(f"{idx+1:02d}. Weight Syn: {r['w_syn']:.1f} | Thresh C0: {r['thresh']:.2f} || Mean Dice: {r['mean_dice']:.4f} | Class 1-9 Dice: {r['class_1_9']:.4f} | Class 0 Dice: {r['class_0']:.4f}")

if __name__ == '__main__':
    main()
