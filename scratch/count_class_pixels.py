import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask

def main():
    mask_paths = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))[:300]
    
    class_pixel_counts = np.zeros(10, dtype=np.int64)
    class_image_counts = np.zeros(10, dtype=np.int64)
    
    for path in tqdm(mask_paths):
        msk = cv2.imread(path)
        msk_rgb = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
        label = rgb_to_mask(msk_rgb, id2color, 10)
        
        for c in range(10):
            pixels = np.sum(label == c)
            class_pixel_counts[c] += pixels
            if pixels > 0:
                class_image_counts[c] += 1
                
    print("\nClass pixel and image frequencies on entire training set:")
    for c in range(10):
        print(f"Class {c:2d}: {class_image_counts[c]:4d} images | Total pixels: {class_pixel_counts[c]:12d} | Avg pixels/img: {class_pixel_counts[c]/len(mask_paths):.1f}")

if __name__ == '__main__':
    main()
