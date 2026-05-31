import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from synergistic_model import FloodNetSynergisticNet
from optimized_pytorch_version import DatasetConfig

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FloodNetSynergisticNet(num_classes=10).to(device)
    
    weights_path = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    state = torch.load(weights_path, map_location=device, weights_only=True)
    state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
    model.load_state_dict(state)
    model.eval()
    
    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))[:100]
    
    # Store probability maps of Class 0 and argmax fallback to save GPU time
    print("Loading test probabilities...")
    class0_probs = []
    fallback_argmax = []
    standard_argmax = []
    
    with torch.no_grad():
        for path in tqdm(test_images):
            base_img = cv2.imread(path)
            orig_h, orig_w = base_img.shape[:2]
            
            img_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            preds_dict = model(img_tensor)
            logits = preds_dict['main_output']
            
            img_flipped = torch.flip(img_tensor, dims=[3])
            preds_flipped_dict = model(img_flipped)
            logits_flipped = preds_flipped_dict['main_output']
            
            probs_std = F.softmax(logits, dim=1)
            probs_flip = F.softmax(logits_flipped, dim=1)
            probs_unflip = torch.flip(probs_flip, dims=[3])
            fused_probs = (probs_std + probs_unflip) * 0.5
            fused_probs_np = fused_probs.squeeze(0).cpu().numpy()
            
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(fused_probs_np[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                
            class0_probs.append(probs_resized[0])
            standard_argmax.append(np.argmax(probs_resized, axis=0))
            fallback_argmax.append(np.argmax(probs_resized[1:], axis=0) + 1)
            
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    
    print("\nTest Set Class 0 Predictions under different thresholds:")
    for thresh in thresholds:
        non_empty_count = 0
        total_pixels = 0
        
        for p0, std_am, fall_am in zip(class0_probs, standard_argmax, fallback_argmax):
            if thresh == 0.0:
                pred_labels = std_am
            elif thresh == 1.0:
                pred_labels = fall_am
            else:
                pred_labels = std_am.copy()
                class0_mask = (pred_labels == 0)
                low_conf_mask = class0_mask & (p0 < thresh)
                if np.any(low_conf_mask):
                    pred_labels[low_conf_mask] = fall_am[low_conf_mask]
            
            c0_pixels = np.sum(pred_labels == 0)
            total_pixels += c0_pixels
            if c0_pixels > 0:
                non_empty_count += 1
                
        print(f"Threshold {thresh:.2f}: {non_empty_count} images have Class 0 | Total Class 0 pixels: {total_pixels}")

if __name__ == '__main__':
    main()
