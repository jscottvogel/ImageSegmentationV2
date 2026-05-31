import os
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from synergistic_model import FloodNetSynergisticNet
from optimized_pytorch_version import DatasetConfig

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

def main():
    device = torch.device('cpu')
    model = FloodNetSynergisticNet(num_classes=10).to(device)
    
    weights_path = "model_checkpoint/FloodNet_Synergistic/best_synergistic_weights.pt"
    state = torch.load(weights_path, map_location=device, weights_only=True)
    state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
    model.load_state_dict(state)
    model.eval()
    
    TEST_IMG_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images"
    test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))[:50]
    
    total_pixels_no_tta_no_resize = 0
    total_pixels_tta_resize_no_morph = 0
    total_pixels_tta_resize_morph50 = 0
    
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
            
            # 1. No TTA, No Resize (directly on logits argmax)
            preds_dict = model(img_tensor)
            logits = preds_dict['main_output']
            pred_labels_no_tta = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            # Resize mask using nearest neighbor to compare
            pred_labels_no_tta_resized = cv2.resize(pred_labels_no_tta, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            total_pixels_no_tta_no_resize += np.sum(pred_labels_no_tta_resized == 0)
            
            # 2. TTA + Resize of probs (no morphological cleanup)
            out_standard_probs = F.softmax(logits, dim=1)
            
            img_flipped = torch.flip(img_tensor, dims=[3])
            preds_flipped_dict = model(img_flipped)
            out_flipped_probs = F.softmax(preds_flipped_dict['main_output'], dim=1)
            out_unflipped_probs = torch.flip(out_flipped_probs, dims=[3])
            
            fused_probs = (out_standard_probs + out_unflipped_probs) * 0.5
            fused_probs_np = fused_probs.squeeze(0).cpu().numpy()
            
            probs_resized = np.zeros((10, orig_h, orig_w), dtype=np.float32)
            for c in range(10):
                probs_resized[c] = cv2.resize(fused_probs_np[c], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            pred_labels_tta = np.argmax(probs_resized, axis=0).astype(np.uint8)
            total_pixels_tta_resize_no_morph += np.sum(pred_labels_tta == 0)
            
            # 3. TTA + Resize of probs + Morphological cleanup (min_area=50)
            pred_labels_tta_morph = safe_morphological_cleanup(pred_labels_tta, min_area=50)
            total_pixels_tta_resize_morph50 += np.sum(pred_labels_tta_morph == 0)
            
    print("\nClass 0 pixel counts on 50 test images:")
    print(f"1. No TTA, argmax on logits, nearest-neighbor resize: {total_pixels_no_tta_no_resize}")
    print(f"2. TTA, bilinear resize on probs, argmax: {total_pixels_tta_resize_no_morph}")
    print(f"3. TTA, bilinear resize on probs, argmax, min_area=50: {total_pixels_tta_resize_morph50}")

if __name__ == '__main__':
    main()
