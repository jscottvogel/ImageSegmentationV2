import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import glob
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2

from synergistic_model import FloodNetSynergisticNet
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask

def safe_morphological_cleanup(pred_labels: np.ndarray, min_area=50) -> np.ndarray:
    if min_area <= 0:
        return pred_labels
    clean_labels = pred_labels.copy()
    for class_id in np.unique(pred_labels):
        if class_id == 0: continue
        class_mask = (pred_labels == class_id).astype(np.uint8)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for i in range(1, num_components):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                clean_labels[labels == i] = 0
    return clean_labels

def evaluate_morphology():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = FloodNetSynergisticNet(num_classes=10).to(device)
    checkpoint_dir = "model_checkpoint/FloodNet_Synergistic"
    best_weights = os.path.join(checkpoint_dir, "best_synergistic_weights.pt")
    
    if os.path.exists(best_weights):
        print(f"Loading weights from: {best_weights}")
        state = torch.load(best_weights, map_location=device, weights_only=True)
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
        model.load_state_dict(state)
        
    model.eval()
    
    tr_img = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    tr_msk = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    np.random.seed(42)
    indices = np.random.choice(len(tr_img), 150, replace=False)
    
    intersection_mc = torch.zeros(10, device=device)
    union_mc = torch.zeros(10, device=device)
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating with Morphology"):
            img = cv2.cvtColor(cv2.imread(tr_img[idx]), cv2.COLOR_BGR2RGB)
            msk = cv2.cvtColor(cv2.imread(tr_msk[idx]), cv2.COLOR_BGR2RGB)
            
            img = cv2.resize(img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            msk = cv2.resize(msk, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            
            label = rgb_to_mask(msk, id2color, 10)
            
            img_tensor = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            target = torch.tensor(label, dtype=torch.long, device=device).unsqueeze(0)
            
            # Prediction
            preds_dict = model(img_tensor)
            out_std = F.softmax(preds_dict['main_output'], dim=1).squeeze(0).cpu().numpy()
            
            # Resize probs back to original size first, then take argmax, then apply morphological cleanup
            orig_h, orig_w = msk.shape[:2] # 480x640 in this validation split
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(out_std[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            pred_labels = np.argmax(probs_resized, axis=0).astype(np.uint8)
            
            # Apply morphological cleanup
            pred_labels = safe_morphological_cleanup(pred_labels, min_area=50)
            
            pred_labels_t = torch.tensor(pred_labels, dtype=torch.long, device=device).unsqueeze(0)
            
            valid_mask = (target != 255)
            target_safe = torch.where(target == 255, torch.zeros_like(target), target)
            
            for c in range(10):
                pred_c = (pred_labels_t == c) & valid_mask
                tgt_c = (target_safe == c) & valid_mask
                
                intersection_mc[c] += torch.sum(pred_c & tgt_c)
                union_mc[c] += torch.sum(pred_c | tgt_c)
                
    dices = ((2. * intersection_mc) / (union_mc + intersection_mc + 1e-6)).cpu().numpy()
    mean_dice = np.mean(dices[1:10])
    print(f"\nMean Dice with Morphology Cleanup (min_area=50): {mean_dice:.4f}")

if __name__ == '__main__':
    evaluate_morphology()
