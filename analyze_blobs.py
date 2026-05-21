import cv2
import glob
import numpy as np
from collections import defaultdict
import os

TRAIN_MASK_DIR = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/train/masks"

id2color = {
    0: [0, 0, 0], 1: [255, 0, 0], 2: [200, 90, 90], 3: [128, 128, 0], 4: [155, 155, 155],
    5: [0, 255, 255], 6: [55, 0, 255], 7: [255, 0, 255], 8: [245, 245, 0], 9: [0, 255, 0],
}
color2id = {tuple(v): k for k, v in id2color.items()}

masks = glob.glob(os.path.join(TRAIN_MASK_DIR, "*.png"))

np.random.seed(42)
if len(masks) > 200:
    masks = np.random.choice(masks, 200, replace=False)

blob_sizes = defaultdict(list)

for mask_path in masks:
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    # Map RGB to class ID
    class_mask_img = np.zeros(mask.shape[:2], dtype=np.uint8)
    for rgb, class_id in color2id.items():
        if class_id == 0: continue
        match = (mask == np.array(rgb)).all(axis=-1)
        
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(match.astype(np.uint8), connectivity=8)
        for i in range(1, num_components):
            area = stats[i, cv2.CC_STAT_AREA]
            blob_sizes[class_id].append(area)

for class_id, sizes in sorted(blob_sizes.items()):
    sizes = np.array(sizes)
    print(f"Class {class_id:2d}: count={len(sizes):5d}, min={sizes.min():5d}, 1st={np.percentile(sizes, 1):5.1f}, 5th={np.percentile(sizes, 5):5.1f}, median={np.median(sizes):6.1f}")
