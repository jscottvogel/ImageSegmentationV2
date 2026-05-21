import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from optimized_pytorch_version import DatasetConfig, CustomDeepLabV3Plus, multiscale_inference
from unet_version import StandardUNet
from fcn_version import ResNet50FCN

id2color = {
    0: [0, 0, 0], 1: [255, 0, 0], 2: [200, 90, 90], 3: [128, 128, 0], 4: [155, 155, 155],
    5: [0, 255, 255], 6: [55, 0, 255], 7: [255, 0, 255], 8: [245, 245, 0], 9: [0, 255, 0],
    255: [255, 255, 255]
}

def decode_segmap(image):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l, color in id2color.items():
        idx = image == l
        r[idx] = color[0]
        g[idx] = color[1]
        b[idx] = color[2]
    # Return as BGR for cv2.imwrite
    return np.stack([b, g, r], axis=2)

# Kaggle Grandmaster Confidence Threshold
# Only pixels where the ensemble is > 95% confident will be labeled.
# Everything else is marked as 255 (Ignore) so the model doesn't memorize mistakes.
CONFIDENCE_THRESHOLD = 0.95 

def generate_confident_pseudo_labels():
    print(f"Generating High-Confidence Pseudo-Labels (Threshold: {CONFIDENCE_THRESHOLD * 100}%)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load DeepLab
    model_cnn = CustomDeepLabV3Plus(num_classes=DatasetConfig.NUM_CLASSES).to(device)
    state_dict_cnn = torch.load('model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt', map_location=device, weights_only=True)
    model_cnn.load_state_dict({k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict_cnn.items()})
    model_cnn.eval()

    # Load UNet
    model_unet = StandardUNet(num_classes=DatasetConfig.NUM_CLASSES).to(device)
    state_dict_unet = torch.load('model_checkpoint/FloodNet_UNet/best_unet_weights.pt', map_location=device, weights_only=True)
    model_unet.load_state_dict({k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict_unet.items()})
    model_unet.eval()

    # Load FCN
    model_fcn = ResNet50FCN(num_classes=DatasetConfig.NUM_CLASSES).to(device)
    state_dict_fcn = torch.load('model_checkpoint/FloodNet_FCN/best_fcn_weights.pt', map_location=device, weights_only=True)
    model_fcn.load_state_dict({k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict_fcn.items()})
    model_fcn.eval()

    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    OUT_DIR = "confident_pseudo_masks"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
    
    with torch.no_grad():
        for img_path in tqdm(test_images):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
            
            # Standardize exactly like the training loop (ImageNet mean/std)
            transform = A.Compose([A.Normalize(), ToTensorV2()])
            augmented = transform(image=img)
            img_tensor = augmented['image'].unsqueeze(0).to(device)
            
            # Generate Probabilities
            prob_cnn = F.softmax(multiscale_inference(model_cnn, img_tensor), dim=1)
            prob_unet = F.softmax(multiscale_inference(model_unet, img_tensor), dim=1)
            prob_fcn = F.softmax(multiscale_inference(model_fcn, img_tensor), dim=1)
            
            # Apply the highly optimized 90.5% weights
            fused_probs = (prob_cnn * 0.42) + (prob_unet * 0.38) + (prob_fcn * 0.20)
            fused_probs = F.interpolate(fused_probs, size=(orig_h, orig_w), mode='bicubic', align_corners=False)
            
            # Extract Maximum Confidence and Labels
            max_probs, pred_labels = torch.max(fused_probs, dim=1)
            max_probs = max_probs.squeeze().cpu().numpy()
            pred_labels = pred_labels.squeeze().cpu().numpy().astype(np.uint8)
            
            # APPLY CONFIDENCE THRESHOLD (The Secret to bypassing the 84% crash)
            pred_labels[max_probs < CONFIDENCE_THRESHOLD] = 255
            
            # Translate the raw 1D matrix into physical RGB colors for the training pipeline
            color_mask = decode_segmap(pred_labels)
            
            out_path = os.path.join(OUT_DIR, os.path.basename(img_path).replace('.jpg', '.png'))
            cv2.imwrite(out_path, color_mask)
            
    print(f"\nSUCCESS! {len(test_images)} High-Confidence Pseudo-Masks generated in '{OUT_DIR}'.")
    print("Next step: Retrain models with these masks mixed into your training directory. The models will ignore the '255' pixels!")

if __name__ == '__main__':
    generate_confident_pseudo_labels()
