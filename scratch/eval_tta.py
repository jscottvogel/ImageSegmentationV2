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

def evaluate_tta():
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
    
    intersection_tta = torch.zeros(10, device=device)
    union_tta = torch.zeros(10, device=device)
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating with TTA"):
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
            
            # TTA Inference
            preds_dict = model(img_tensor)
            out_std = F.softmax(preds_dict['main_output'], dim=1)
            
            img_flipped = torch.flip(img_tensor, dims=[3])
            preds_dict_flipped = model(img_flipped)
            out_flipped = F.softmax(preds_dict_flipped['main_output'], dim=1)
            out_unflipped = torch.flip(out_flipped, dims=[3])
            
            fused = (out_std + out_unflipped) * 0.5
            pred_labels = torch.argmax(fused, dim=1)
            
            valid_mask = (target != 255)
            target_safe = torch.where(target == 255, torch.zeros_like(target), target)
            
            for c in range(10):
                pred_c = (pred_labels == c) & valid_mask
                tgt_c = (target_safe == c) & valid_mask
                
                intersection_tta[c] += torch.sum(pred_c & tgt_c)
                union_tta[c] += torch.sum(pred_c | tgt_c)
                
    dices = ((2. * intersection_tta) / (union_tta + intersection_tta + 1e-6)).cpu().numpy()
    mean_dice = np.mean(dices[1:10])
    print(f"\nMean Dice with Horizontal Flip TTA: {mean_dice:.4f}")

if __name__ == '__main__':
    evaluate_tta()
