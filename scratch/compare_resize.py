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
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))[:50]
    
    total_pixels_probs = 0
    total_images_probs = 0
    
    total_pixels_logits = 0
    total_images_logits = 0
    
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
            
            # Predict
            preds_dict = model(img_tensor)
            logits_std = preds_dict['main_output']
            
            # TTA horizontal flip
            img_flipped = torch.flip(img_tensor, dims=[3])
            preds_flipped_dict = model(img_flipped)
            logits_flipped = preds_flipped_dict['main_output']
            
            # Option A: Resizing probabilities (current inference_synergistic.py logic)
            probs_std = F.softmax(logits_std, dim=1)
            probs_flipped = F.softmax(logits_flipped, dim=1)
            probs_unflipped = torch.flip(probs_flipped, dims=[3])
            fused_probs = (probs_std + probs_unflipped) * 0.5
            fused_probs_np = fused_probs.squeeze(0).cpu().numpy()
            
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(fused_probs_np[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            labels_probs = np.argmax(probs_resized, axis=0)
            
            p_count = np.sum(labels_probs == 0)
            total_pixels_probs += p_count
            if p_count > 0:
                total_images_probs += 1
                
            # Option B: Resizing logits (run_final_submission.py logic)
            # Unflip logits flipped
            logits_unflipped = torch.flip(logits_flipped, dims=[3])
            fused_logits = (logits_std + logits_unflipped) * 0.5
            fused_logits_np = fused_logits.squeeze(0).cpu().numpy()
            
            logits_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                logits_resized[c] = cv2.resize(fused_logits_np[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            labels_logits = np.argmax(logits_resized, axis=0)
            
            l_count = np.sum(labels_logits == 0)
            total_pixels_logits += l_count
            if l_count > 0:
                total_images_logits += 1
                
    print("\nComparison on first 50 test images:")
    print(f"Resizing Probabilities | Images with Class 0: {total_images_probs} | Total Class 0 pixels: {total_pixels_probs}")
    print(f"Resizing Logits        | Images with Class 0: {total_images_logits} | Total Class 0 pixels: {total_pixels_logits}")

if __name__ == '__main__':
    main()
