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

def evaluate_synergistic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the synergistic model
    model = FloodNetSynergisticNet(num_classes=10).to(device)
    
    # Select weights
    checkpoint_dir = "model_checkpoint/FloodNet_Synergistic"
    swa_weights = os.path.join(checkpoint_dir, "final_swa_smoothed_synergistic.pt")
    best_weights = os.path.join(checkpoint_dir, "best_synergistic_weights.pt")
    
    weights_path = None
    if os.path.exists(best_weights):
        weights_path = best_weights
    elif os.path.exists(swa_weights):
        weights_path = swa_weights
    else:
        print("[WARNING] No trained weights found! Evaluating with random weights.")
        
    if weights_path:
        print(f"Loading weights from: {weights_path}")
        state = torch.load(weights_path, map_location=device, weights_only=True)
        # Clean state dict keys if they contain prefixes
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
        model.load_state_dict(state, strict=False)
        
    model.eval()
    
    # 2. Get validation data paths
    # We will use the same indices/dataset structure
    tr_img = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    tr_msk = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    np.random.seed(42)
    indices = np.random.choice(len(tr_img), 150, replace=False)
    
    # Intersection & Union counters for each output head
    heads = ['main_output', 'unet_output', 'deeplab_output', 'fcn_output']
    intersections = {h: torch.zeros(10, device=device) for h in heads}
    unions = {h: torch.zeros(10, device=device) for h in heads}
    
    # Evaluation loop
    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating Synergistic Net"):
            img = cv2.cvtColor(cv2.imread(tr_img[idx]), cv2.COLOR_BGR2RGB)
            msk = cv2.cvtColor(cv2.imread(tr_msk[idx]), cv2.COLOR_BGR2RGB)
            
            img = cv2.resize(img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            msk = cv2.resize(msk, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            
            label = rgb_to_mask(msk, id2color, 10)
            
            # Preprocessing
            img_tensor = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            target = torch.tensor(label, dtype=torch.long, device=device).unsqueeze(0)
            
            # Forward pass
            preds_dict = model(img_tensor)
            
            valid_mask = (target != 255)
            target_safe = torch.where(target == 255, torch.zeros_like(target), target)
            
            for h in heads:
                out = preds_dict[h]
                pred_labels = torch.argmax(out, dim=1)
                
                for c in range(10):
                    pred_c = (pred_labels == c) & valid_mask
                    tgt_c = (target_safe == c) & valid_mask
                    
                    intersections[h][c] += torch.sum(pred_c & tgt_c)
                    unions[h][c] += torch.sum(pred_c | tgt_c)
                    
    # Print results
    print("\n" + "="*50)
    print("SYNERGISTIC DECODER HEADS CLASS-WISE HARD DICE")
    print("="*50)
    
    dices = {}
    for h in heads:
        dices[h] = ((2. * intersections[h]) / (unions[h] + intersections[h] + 1e-6)).cpu().numpy()
        
    print(f"{'Class':8s} | {'Fused Main':10s} | {'UNet Head':10s} | {'DeepLab Head':12s} | {'FCN Head':10s}")
    for c in range(1, 10):
        print(f"Class {c:2d}     | {dices['main_output'][c]:.4f}     | {dices['unet_output'][c]:.4f}    | {dices['deeplab_output'][c]:.4f}       | {dices['fcn_output'][c]:.4f}")
        
    print("-"*50)
    for h in heads:
        mean_dice = np.mean(dices[h][1:10]) # Mean Dice over non-background classes
        print(f"Mean Dice ({h}): {mean_dice:.4f}")
    print("="*50)

if __name__ == '__main__':
    evaluate_synergistic()
