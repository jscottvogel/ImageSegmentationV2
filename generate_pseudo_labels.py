import os
import cv2
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

def decode_rle_to_mask(rle_string, height, width):
    """Decodes Kaggle RLE string back into a 2D dense binary matrix."""
    mask = np.zeros(height * width, dtype=np.uint8)
    if pd.isna(rle_string) or rle_string == '':
        return mask.reshape((width, height)).T
    
    runs = np.array([int(x) for x in rle_string.split()])
    starts = runs[0::2] - 1
    lengths = runs[1::2]
    
    for start, length in zip(starts, lengths):
        mask[start:start+length] = 1
        
    return mask.reshape((width, height)).T

def generate_pseudo_masks(submission_csv="hybrid_ensemble_submission.csv", 
                          test_img_dir="/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/test/images",
                          output_dir="pseudo_masks"):
    """
    Transforms the highly accurate 90.28% CSV predictions back into physical .png masks.
    These can be mixed with the training data to force the models to adapt to the Kaggle domain shift.
    """
    print(f"Reading High-Accuracy Ensemble CSV: {submission_csv}")
    df = pd.read_csv(submission_csv)
    
    os.makedirs(output_dir, exist_ok=True)
    test_images = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
    
    print("Extracting physical masks from mathematical RLE sequences...")
    for img_path in tqdm(test_images):
        filename = os.path.basename(img_path)
        base_name = filename.replace('.jpg', '')
        
        # Load image strictly to extract the raw physical dimensions
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Reconstruct the 10-class segmentation map layer by layer
        for class_id in range(1, 10):
            img_id_str = f"{base_name}_{class_id:02d}"
            match = df[df['IMG_ID'] == img_id_str]
            if not match.empty:
                rle = match.iloc[0]['EncodedString']
                binary_mask = decode_rle_to_mask(rle, h, w)
                final_mask[binary_mask == 1] = class_id
                
        # Save pseudo-mask
        out_path = os.path.join(output_dir, base_name + ".png")
        cv2.imwrite(out_path, final_mask)

    print(f"\nSUCCESS! {len(test_images)} Pseudo-Masks mathematically generated at: {output_dir}")
    print("Next Step: Point your DatasetConfig.TRAIN_MSK_DIR to include these masks and retrain to capture the Domain Shift!")

if __name__ == '__main__':
    generate_pseudo_masks()
