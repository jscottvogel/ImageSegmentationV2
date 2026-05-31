import os
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from synergistic_model import FloodNetSynergisticNet
from optimized_pytorch_version import DatasetConfig

def main():
    device = torch.device('cpu')
    model = FloodNetSynergisticNet(num_classes=10).to(device)
    
    weights_path = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    state = torch.load(weights_path, map_location=device, weights_only=True)
    state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
    model.load_state_dict(state)
    model.eval()
    
    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))[:50] # Check first 50 images
    
    counts = {
        'main': 0,
        'unet': 0,
        'deeplab': 0,
        'fcn': 0
    }
    
    pixels = {
        'main': 0,
        'unet': 0,
        'deeplab': 0,
        'fcn': 0
    }
    
    with torch.no_grad():
        for path in tqdm(test_images):
            base_img = cv2.imread(path)
            img_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            
            img_tensor = img_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.tensor(img_tensor.transpose(2,0,1)[None, ...], dtype=torch.float32).to(device)
            
            preds_dict = model(img_tensor)
            
            for key in ['main', 'unet', 'deeplab', 'fcn']:
                if key == 'main':
                    logits = preds_dict['main_output']
                elif key == 'unet':
                    logits = preds_dict['unet_output']
                elif key == 'deeplab':
                    logits = preds_dict['deeplab_output']
                elif key == 'fcn':
                    logits = preds_dict['fcn_output']
                    
                pred_labels = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
                class0_pixels = np.sum(pred_labels == 0)
                pixels[key] += class0_pixels
                if class0_pixels > 0:
                    counts[key] += 1
                    
    print("\nResults on first 50 test images:")
    for key in ['main', 'unet', 'deeplab', 'fcn']:
        print(f"Head {key:8s}: {counts[key]} images have Class 0 | Total Class 0 pixels: {pixels[key]}")

if __name__ == '__main__':
    main()
