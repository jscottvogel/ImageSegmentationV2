import pandas as pd
import numpy as np
from tqdm import tqdm

def rle2mask(rle_str, shape=(3000, 4000)):
    # 1-based RLE decoding
    h, w = shape
    mask = np.zeros(h * w, dtype=np.uint8)
    if not isinstance(rle_str, str) or rle_str.strip() == "":
        return mask
    s = rle_str.split()
    starts = np.asarray(s[0::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)
    for start, length in zip(starts, lengths):
        mask[start:start+length] = 1
    return mask

def main():
    import sys
    csv_path = "final_competitive_submission.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Each row is IMG_ID (e.g. 10008_00) and EncodedString
    class_pixel_counts = np.zeros(10, dtype=np.int64)
    class_image_counts = np.zeros(10, dtype=np.int64)
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['IMG_ID']
        rle = row['EncodedString']
        class_id = int(img_id.split('_')[-1])
        
        if pd.isna(rle) or not isinstance(rle, str) or rle.strip() == "":
            continue
            
        # Let's count runs lengths to avoid full array decoding for speed
        s = rle.split()
        lengths = np.asarray(s[1::2], dtype=int)
        pixels = np.sum(lengths)
        
        if pixels > 0:
            class_pixel_counts[class_id] += pixels
            class_image_counts[class_id] += 1
            
    print(f"\nClass prediction statistics for {csv_path}:")
    for c in range(10):
        print(f"Class {c:2d}: {class_image_counts[c]:4d} images | Total pixels: {class_pixel_counts[c]:12d}")

if __name__ == '__main__':
    main()
