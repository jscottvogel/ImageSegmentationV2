import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"
import cv2
import torch
import glob
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from optimized_pytorch_version import CustomDeepLabV3Plus, DatasetConfig, rgb_to_mask
from unet_version import StandardUNet
from fcn_version import ResNet50FCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
id2color = {
    0: [0, 0, 0], 1: [255, 0, 0], 2: [200, 90, 90], 3: [128, 128, 0], 4: [155, 155, 155],
    5: [0, 255, 255], 6: [55, 0, 255], 7: [255, 0, 255], 8: [245, 245, 0], 9: [0, 255, 0],
    255: [255, 255, 255]
}

def normalize_imagenet(img_tensor, device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (img_tensor - mean) / std

def load_weights(model, path):
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        return None
    state = torch.load(path, map_location=device, weights_only=True)
    if 'n_averaged' in state: del state['n_averaged']
    clean_state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    return model

image_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))

_, val_images, _, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

# Use 150 validation images for a rigorous check (same random split)
np.random.seed(42)
indices = np.random.choice(len(val_images), 150, replace=False)
val_images = [val_images[idx] for idx in indices]
val_masks = [val_masks[idx] for idx in indices]
print(f"Loaded {len(val_images)} validation images")

models = {
    'DeepLab Best Active': (CustomDeepLabV3Plus(num_classes=10).to(device), 'model_checkpoint/FloodNet_PyTorch/best_deeplab_weights.pt'),
    'DeepLab SWA Active': (CustomDeepLabV3Plus(num_classes=10).to(device), 'model_checkpoint/FloodNet_PyTorch/final_swa_smoothed_weights.pt'),
    'UNet Best Active': (StandardUNet(num_classes=10).to(device), 'model_checkpoint/FloodNet_UNet/best_unet_weights.pt'),
    'UNet SWA Active': (StandardUNet(num_classes=10).to(device), 'model_checkpoint/FloodNet_UNet/final_swa_smoothed_unet.pt'),
    'FCN Best Active': (ResNet50FCN(num_classes=10).to(device), 'model_checkpoint/FloodNet_FCN/best_fcn_weights.pt'),
    'FCN SWA Active': (ResNet50FCN(num_classes=10).to(device), 'model_checkpoint/FloodNet_FCN/final_swa_smoothed_fcn.pt')
}

for name, (model_obj, path) in models.items():
    model = load_weights(model_obj, path)
    if model is None:
        continue
    intersection = np.zeros(10)
    union = np.zeros(10)
    with torch.no_grad():
        for i in range(len(val_images)):
            img = cv2.cvtColor(cv2.imread(val_images[i]), cv2.COLOR_BGR2RGB)
            msk = cv2.cvtColor(cv2.imread(val_masks[i]), cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
            label = rgb_to_mask(msk, id2color, 10)
            
            img_res = cv2.resize(img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT))
            img_t = torch.tensor(img_res.transpose(2, 0, 1)[None, ...], dtype=torch.float32, device=device) / 255.0
            norm_img = normalize_imagenet(img_t, device)
            
            out = model(norm_img)
            if isinstance(out, dict): out = out['main_output']
            out_res = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze(0)
            pred = out_res.argmax(dim=0).cpu().numpy()
            
            valid = (label != 255)
            for c in range(10):
                pred_c = (pred == c) & valid
                tgt_c = (label == c) & valid
                intersection[c] += np.sum(pred_c & tgt_c)
                union[c] += np.sum(pred_c | tgt_c)
                
    dice = (2. * intersection) / (union + intersection + 1e-6)
    print(f'{name} | Class 1-9 Avg Dice: {np.mean(dice[1:]):.4f}')
