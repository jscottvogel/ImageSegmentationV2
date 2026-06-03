import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from synergistic_model import FloodNetSynergisticNet
from competitive_model import FloodNetCompetitiveModel
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask
from scratch.eval_advanced_tta import neighbor_fill_cleanup

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading models...")
    
    model_syn = FloodNetSynergisticNet(num_classes=10).to(device)
    syn_weights = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    state = torch.load(syn_weights, map_location=device, weights_only=True)
    state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
    model_syn.load_state_dict(state)
    model_syn.eval()
    
    model_meta = FloodNetCompetitiveModel(num_classes=10).to(device)
    model_meta.load_checkpoints(
        unet_path="model_checkpoint/FloodNet_UNet/best_unet_weights.pt",
        fcn_path="model_checkpoint/FloodNet_FCN/best_fcn_weights.pt",
        deeplab_path="model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt",
        meta_path="model_checkpoint/FloodNet_Meta/meta_layer_weights.pt",
        device=device
    )
    model_meta.eval()
    
    tr_img = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    tr_msk = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    np.random.seed(42)
    indices = np.random.choice(len(tr_img), 30, replace=False)
    
    print("\nRunning Evaluation at High-Res (original method: resize probs to 3000x4000, argmax, cleanup)...")
    class_intersections_hr = np.zeros(10, dtype=np.float64)
    class_unions_hr = np.zeros(10, dtype=np.float64)
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="High-Res Eval"):
            img_bgr = cv2.imread(tr_img[idx])
            orig_h, orig_w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Ground truth at original resolution
            gt_rgb = cv2.cvtColor(cv2.imread(tr_msk[idx]), cv2.COLOR_BGR2RGB)
            gt_label = rgb_to_mask(gt_rgb, id2color, 10)
            
            # Predict
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            out_syn = model_syn(img_tensor)
            p_syn_main = F.softmax(out_syn['main_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_unet = F.softmax(out_syn['unet_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_dl = F.softmax(out_syn['deeplab_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_notta = 0.4 * p_syn_main + 0.3 * p_syn_unet + 0.3 * p_syn_dl
            p_syn_notta[0] = p_syn_main[0]
            
            meta_logits = model_meta(img_tensor)
            p_meta = F.softmax(meta_logits, dim=1).cpu().squeeze(0).numpy()
            
            p_blend = 0.50 * p_syn_notta + 0.50 * p_meta
            p_blend[0] = p_syn_notta[0]
            
            # Resize probs to high resolution
            probs_hr = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_hr[c] = cv2.resize(p_blend[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                
            pred_labels = np.argmax(probs_hr, axis=0).astype(np.uint8)
            low_conf = (pred_labels == 0) & (probs_hr[0] < 0.90)
            if np.any(low_conf):
                fallback = np.argmax(probs_hr[1:], axis=0) + 1
                pred_labels[low_conf] = fallback[low_conf]
                
            pred_labels = neighbor_fill_cleanup(pred_labels, min_area=50)
            
            valid_mask = (gt_label != 255)
            for c in range(10):
                pred_c = (pred_labels == c) & valid_mask
                gt_c = (gt_label == c) & valid_mask
                class_intersections_hr[c] += np.sum(pred_c & gt_c)
                class_unions_hr[c] += np.sum(pred_c) + np.sum(gt_c)
                
    mean_dice_hr = []
    for c in range(1, 10):
        dice = (2. * class_intersections_hr[c]) / (class_unions_hr[c] + 1e-6)
        mean_dice_hr.append(dice)
    print(f"High-Res Mean Dice: {np.mean(mean_dice_hr):.6f}")
    
    print("\nRunning Evaluation at Low-Res (fast method: argmax and cleanup at 480x640, then nearest resize mask)...")
    class_intersections_lr = np.zeros(10, dtype=np.float64)
    class_unions_lr = np.zeros(10, dtype=np.float64)
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Low-Res Eval"):
            img_bgr = cv2.imread(tr_img[idx])
            orig_h, orig_w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # Ground truth at original resolution
            gt_rgb = cv2.cvtColor(cv2.imread(tr_msk[idx]), cv2.COLOR_BGR2RGB)
            gt_label = rgb_to_mask(gt_rgb, id2color, 10)
            
            # Predict
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            out_syn = model_syn(img_tensor)
            p_syn_main = F.softmax(out_syn['main_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_unet = F.softmax(out_syn['unet_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_dl = F.softmax(out_syn['deeplab_output'], dim=1).cpu().squeeze(0).numpy()
            p_syn_notta = 0.4 * p_syn_main + 0.3 * p_syn_unet + 0.3 * p_syn_dl
            p_syn_notta[0] = p_syn_main[0]
            
            meta_logits = model_meta(img_tensor)
            p_meta = F.softmax(meta_logits, dim=1).cpu().squeeze(0).numpy()
            
            p_blend = 0.50 * p_syn_notta + 0.50 * p_meta
            p_blend[0] = p_syn_notta[0]
            
            # Argmax and cleanup at low res (480x640)
            pred_labels_lr = np.argmax(p_blend, axis=0).astype(np.uint8)
            low_conf = (pred_labels_lr == 0) & (p_blend[0] < 0.90)
            if np.any(low_conf):
                fallback = np.argmax(p_blend[1:], axis=0) + 1
                pred_labels_lr[low_conf] = fallback[low_conf]
                
            pred_labels_lr = neighbor_fill_cleanup(pred_labels_lr, min_area=2)
            
            # Resize clean mask to original resolution using INTER_NEAREST
            pred_labels = cv2.resize(pred_labels_lr, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            valid_mask = (gt_label != 255)
            for c in range(10):
                pred_c = (pred_labels == c) & valid_mask
                gt_c = (gt_label == c) & valid_mask
                class_intersections_lr[c] += np.sum(pred_c & gt_c)
                class_unions_lr[c] += np.sum(pred_c) + np.sum(gt_c)
                
    mean_dice_lr = []
    for c in range(1, 10):
        dice = (2. * class_intersections_lr[c]) / (class_unions_lr[c] + 1e-6)
        mean_dice_lr.append(dice)
    print(f"Low-Res + Nearest Resize Mean Dice: {np.mean(mean_dice_lr):.6f} (Delta: {np.mean(mean_dice_lr) - np.mean(mean_dice_hr):+.6f})")

if __name__ == '__main__':
    main()
