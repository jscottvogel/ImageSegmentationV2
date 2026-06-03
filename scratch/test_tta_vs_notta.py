import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from synergistic_model import FloodNetSynergisticNet
from competitive_model import FloodNetCompetitiveModel
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask
from scratch.eval_advanced_tta import neighbor_fill_cleanup

def apply_multiclass_thresholding(probs: np.ndarray, thresh_dict: dict) -> np.ndarray:
    pred = np.argmax(probs, axis=0)
    non_suppressed_classes = [c for c in range(10) if c not in thresh_dict or thresh_dict[c] == 0]
    fallback_idx = np.argmax(probs[non_suppressed_classes], axis=0)
    fallback = np.array(non_suppressed_classes)[fallback_idx]
    
    for c, t in thresh_dict.items():
        if t > 0:
            mask = (pred == c) & (probs[c] < t)
            pred[mask] = fallback[mask]
    return pred

def evaluate_predictions(gt_labels, pred_labels_list):
    class_intersections = np.zeros(10, dtype=np.float64)
    class_unions = np.zeros(10, dtype=np.float64)
    
    for gt, pred in zip(gt_labels, pred_labels_list):
        valid_mask = (gt != 255)
        for c in range(10):
            pred_c = (pred == c) & valid_mask
            gt_c = (gt == c) & valid_mask
            class_intersections[c] += np.sum(pred_c & gt_c)
            class_unions[c] += np.sum(pred_c) + np.sum(gt_c)
            
    mean_dice_1_9 = []
    dice_per_class = {}
    for c in range(1, 10):
        dice = (2. * class_intersections[c]) / (class_unions[c] + 1e-6)
        mean_dice_1_9.append(dice)
        dice_per_class[c] = dice
        
    return np.mean(mean_dice_1_9), dice_per_class

def get_predictions(model_syn, model_meta, x, device):
    # Predict Synergistic Net
    out_syn = model_syn(x)
    p_syn_main = F.softmax(out_syn['main_output'], dim=1)
    p_syn_unet = F.softmax(out_syn['unet_output'], dim=1)
    p_syn_dl = F.softmax(out_syn['deeplab_output'], dim=1)
    
    # Predict Meta Net
    meta_logits = model_meta(x)
    p_meta = F.softmax(meta_logits, dim=1)
    
    return p_syn_main, p_syn_unet, p_syn_dl, p_meta

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Models
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
    
    image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    _, val_image_paths, _, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    np.random.seed(123)
    indices = np.random.choice(len(val_image_paths), 30, replace=False)
    val_sub_images = [val_image_paths[i] for i in indices]
    val_sub_masks = [val_mask_paths[i] for i in indices]
    
    gts = []
    imgs_rgb = []
    for idx in range(len(val_sub_images)):
        img = cv2.cvtColor(cv2.imread(val_sub_images[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.cvtColor(cv2.imread(val_sub_masks[idx]), cv2.COLOR_BGR2RGB)
        gt = rgb_to_mask(msk, id2color, 10)
        gts.append(gt)
        imgs_rgb.append(img)
        
    thresh_dict = {0: 0.95, 3: 0.50, 1: 0.50}
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # Pre-load/normalize inputs
    img_tensors = []
    for img in imgs_rgb:
        img_norm = (img.astype(np.float32) / 255.0 - mean) / std
        img_tensor = torch.tensor(img_norm.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
        img_tensors.append(img_tensor)
        
    # ------------------ 1. NO TTA ------------------
    preds_no_tta = []
    with torch.no_grad():
        for x in img_tensors:
            p_syn_main, p_syn_unet, p_syn_dl, p_meta = get_predictions(model_syn, model_meta, x, device)
            p_syn_main = p_syn_main.cpu().squeeze(0).numpy()
            p_syn = (0.4 * p_syn_main + 
                     0.3 * p_syn_unet.cpu().squeeze(0).numpy() + 
                     0.3 * p_syn_dl.cpu().squeeze(0).numpy())
            p_syn[0] = p_syn_main[0]
            
            p_m = p_meta.cpu().squeeze(0).numpy()
            p_blend = 0.50 * p_syn + 0.50 * p_m
            p_blend[0] = p_syn[0]
            
            pred_labels = apply_multiclass_thresholding(p_blend, thresh_dict)
            pred_labels = neighbor_fill_cleanup(pred_labels.astype(np.uint8), min_area=128)
            preds_no_tta.append(pred_labels)
            
    dice_no_tta, dice_no_tta_class = evaluate_predictions(gts, preds_no_tta)
    print(f"NO TTA -> Macro Dice: {dice_no_tta:.6f} | C1: {dice_no_tta_class[1]:.6f} | C3: {dice_no_tta_class[3]:.6f}")

    # ------------------ 2. H-FLIP TTA ------------------
    preds_h_flip = []
    with torch.no_grad():
        for x in img_tensors:
            # Std pass
            p_syn_main, p_syn_unet, p_syn_dl, p_meta = get_predictions(model_syn, model_meta, x, device)
            
            # Flipped pass
            x_flip = torch.flip(x, dims=[3])
            pf_syn_main, pf_syn_unet, pf_syn_dl, pf_meta = get_predictions(model_syn, model_meta, x_flip, device)
            
            # Unflip predictions
            pf_syn_main = torch.flip(pf_syn_main, dims=[3])
            pf_syn_unet = torch.flip(pf_syn_unet, dims=[3])
            pf_syn_dl = torch.flip(pf_syn_dl, dims=[3])
            pf_meta = torch.flip(pf_meta, dims=[3])
            
            # Average
            p_syn_main_avg = ((p_syn_main + pf_syn_main) * 0.5).cpu().squeeze(0).numpy()
            p_syn_unet_avg = ((p_syn_unet + pf_syn_unet) * 0.5).cpu().squeeze(0).numpy()
            p_syn_dl_avg = ((p_syn_dl + pf_syn_dl) * 0.5).cpu().squeeze(0).numpy()
            p_meta_avg = ((p_meta + pf_meta) * 0.5).cpu().squeeze(0).numpy()
            
            p_syn = 0.4 * p_syn_main_avg + 0.3 * p_syn_unet_avg + 0.3 * p_syn_dl_avg
            p_syn[0] = p_syn_main_avg[0]
            
            p_blend = 0.50 * p_syn + 0.50 * p_meta_avg
            p_blend[0] = p_syn[0]
            
            pred_labels = apply_multiclass_thresholding(p_blend, thresh_dict)
            pred_labels = neighbor_fill_cleanup(pred_labels.astype(np.uint8), min_area=128)
            preds_h_flip.append(pred_labels)
            
    dice_h_flip, dice_h_flip_class = evaluate_predictions(gts, preds_h_flip)
    print(f"H-FLIP TTA -> Macro Dice: {dice_h_flip:.6f} (Delta: {dice_h_flip - dice_no_tta:+.6f})")

    # ------------------ 3. V-FLIP TTA ------------------
    preds_v_flip = []
    with torch.no_grad():
        for x in img_tensors:
            # Std pass
            p_syn_main, p_syn_unet, p_syn_dl, p_meta = get_predictions(model_syn, model_meta, x, device)
            
            # Flipped pass
            x_flip = torch.flip(x, dims=[2])
            pf_syn_main, pf_syn_unet, pf_syn_dl, pf_meta = get_predictions(model_syn, model_meta, x_flip, device)
            
            # Unflip predictions
            pf_syn_main = torch.flip(pf_syn_main, dims=[2])
            pf_syn_unet = torch.flip(pf_syn_unet, dims=[2])
            pf_syn_dl = torch.flip(pf_syn_dl, dims=[2])
            pf_meta = torch.flip(pf_meta, dims=[2])
            
            # Average
            p_syn_main_avg = ((p_syn_main + pf_syn_main) * 0.5).cpu().squeeze(0).numpy()
            p_syn_unet_avg = ((p_syn_unet + pf_syn_unet) * 0.5).cpu().squeeze(0).numpy()
            p_syn_dl_avg = ((p_syn_dl + pf_syn_dl) * 0.5).cpu().squeeze(0).numpy()
            p_meta_avg = ((p_meta + pf_meta) * 0.5).cpu().squeeze(0).numpy()
            
            p_syn = 0.4 * p_syn_main_avg + 0.3 * p_syn_unet_avg + 0.3 * p_syn_dl_avg
            p_syn[0] = p_syn_main_avg[0]
            
            p_blend = 0.50 * p_syn + 0.50 * p_meta_avg
            p_blend[0] = p_syn[0]
            
            pred_labels = apply_multiclass_thresholding(p_blend, thresh_dict)
            pred_labels = neighbor_fill_cleanup(pred_labels.astype(np.uint8), min_area=128)
            preds_v_flip.append(pred_labels)
            
    dice_v_flip, dice_v_flip_class = evaluate_predictions(gts, preds_v_flip)
    print(f"V-FLIP TTA -> Macro Dice: {dice_v_flip:.6f} (Delta: {dice_v_flip - dice_no_tta:+.6f})")

    # ------------------ 4. FULL (H + V) FLIP TTA ------------------
    preds_full_flip = []
    with torch.no_grad():
        for x in img_tensors:
            p_syn_main, p_syn_unet, p_syn_dl, p_meta = get_predictions(model_syn, model_meta, x, device)
            
            # H-Flip
            x_h = torch.flip(x, dims=[3])
            ph_syn_main, ph_syn_unet, ph_syn_dl, ph_meta = get_predictions(model_syn, model_meta, x_h, device)
            ph_syn_main = torch.flip(ph_syn_main, dims=[3])
            ph_syn_unet = torch.flip(ph_syn_unet, dims=[3])
            ph_syn_dl = torch.flip(ph_syn_dl, dims=[3])
            ph_meta = torch.flip(ph_meta, dims=[3])
            
            # V-Flip
            x_v = torch.flip(x, dims=[2])
            pv_syn_main, pv_syn_unet, pv_syn_dl, pv_meta = get_predictions(model_syn, model_meta, x_v, device)
            pv_syn_main = torch.flip(pv_syn_main, dims=[2])
            pv_syn_unet = torch.flip(pv_syn_unet, dims=[2])
            pv_syn_dl = torch.flip(pv_syn_dl, dims=[2])
            pv_meta = torch.flip(pv_meta, dims=[2])
            
            # HV-Flip
            x_hv = torch.flip(x, dims=[2, 3])
            phv_syn_main, phv_syn_unet, phv_syn_dl, phv_meta = get_predictions(model_syn, model_meta, x_hv, device)
            phv_syn_main = torch.flip(phv_syn_main, dims=[2, 3])
            phv_syn_unet = torch.flip(phv_syn_unet, dims=[2, 3])
            phv_syn_dl = torch.flip(phv_syn_dl, dims=[2, 3])
            phv_meta = torch.flip(phv_meta, dims=[2, 3])
            
            # Average
            p_syn_main_avg = ((p_syn_main + ph_syn_main + pv_syn_main + phv_syn_main) * 0.25).cpu().squeeze(0).numpy()
            p_syn_unet_avg = ((p_syn_unet + ph_syn_unet + pv_syn_unet + phv_syn_unet) * 0.25).cpu().squeeze(0).numpy()
            p_syn_dl_avg = ((p_syn_dl + ph_syn_dl + pv_syn_dl + phv_syn_dl) * 0.25).cpu().squeeze(0).numpy()
            p_meta_avg = ((p_meta + ph_meta + pv_meta + phv_meta) * 0.25).cpu().squeeze(0).numpy()
            
            p_syn = 0.4 * p_syn_main_avg + 0.3 * p_syn_unet_avg + 0.3 * p_syn_dl_avg
            p_syn[0] = p_syn_main_avg[0]
            
            p_blend = 0.50 * p_syn + 0.50 * p_meta_avg
            p_blend[0] = p_syn[0]
            
            pred_labels = apply_multiclass_thresholding(p_blend, thresh_dict)
            pred_labels = neighbor_fill_cleanup(pred_labels.astype(np.uint8), min_area=128)
            preds_full_flip.append(pred_labels)
            
    dice_full_flip, dice_full_flip_class = evaluate_predictions(gts, preds_full_flip)
    print(f"FULL FLIP TTA -> Macro Dice: {dice_full_flip:.6f} (Delta: {dice_full_flip - dice_no_tta:+.6f})")

if __name__ == '__main__':
    main()
