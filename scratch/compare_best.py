import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# RLE Decoder
def rle2mask(rle_str, length=307200):
    if not rle_str or pd.isna(rle_str) or rle_str == "":
        return np.zeros(length, dtype=bool)
    s = [int(x) for x in rle_str.split()]
    starts, lengths = s[::2], s[1::2]
    mask = np.zeros(length, dtype=bool)
    for start, l in zip(starts, lengths):
        mask[start:start+l] = True
    return mask

def compare_two(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    df1 = df1.sort_values("IMG_ID").reset_index(drop=True)
    df2 = df2.sort_values("IMG_ID").reset_index(drop=True)
    
    total_intersection = 0
    total_sum1 = 0
    total_sum2 = 0
    
    class_intersections = {c: 0 for c in range(10)}
    class_sums1 = {c: 0 for c in range(10)}
    class_sums2 = {c: 0 for c in range(10)}
    
    for idx, row in tqdm(df1.iterrows(), total=len(df1)):
        img_id = row["IMG_ID"]
        class_id = int(img_id.split("_")[1])
        
        rle1 = row["EncodedString"]
        rle2 = df2.loc[idx, "EncodedString"]
        
        mask1 = rle2mask(rle1)
        mask2 = rle2mask(rle2)
        
        intersection = np.sum(mask1 & mask2)
        sum1 = np.sum(mask1)
        sum2 = np.sum(mask2)
        
        total_intersection += intersection
        total_sum1 += sum1
        total_sum2 += sum2
        
        class_intersections[class_id] += intersection
        class_sums1[class_id] += sum1
        class_sums2[class_id] += sum2
        
    global_dice = (2.0 * total_intersection) / (total_sum1 + total_sum2 + 1e-6)
    print(f"\nGlobal Dice Overlap: {global_dice:.6f}")
    
    print("\nClass-wise Dice Overlap:")
    for c in range(10):
        dice = (2.0 * class_intersections[c]) / (class_sums1[c] + class_sums2[c] + 1e-6)
        print(f"  Class {c}: {dice:.6f} (Pixel Count in File 1: {class_sums1[c]}, File 2: {class_sums2[c]})")

if __name__ == '__main__':
    f1 = "ensemble_w50_t95_c3t50_c1t50_area128_submission.csv"
    f2 = "final_kaggle_submission.csv"
    compare_two(f1, f2)
