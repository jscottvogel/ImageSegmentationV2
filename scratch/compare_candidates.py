import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def rle2mask(rle_str, length=12000000):
    if not rle_str or pd.isna(rle_str) or rle_str == "":
        return np.zeros(length, dtype=bool)
    s = [int(x) for x in rle_str.split()]
    starts, lengths = s[::2], s[1::2]
    mask = np.zeros(length, dtype=bool)
    for start, l in zip(starts, lengths):
        mask[start-1:start-1+l] = True
    return mask

def compare_submissions(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    df1 = df1.sort_values("IMG_ID").reset_index(drop=True)
    df2 = df2.sort_values("IMG_ID").reset_index(drop=True)
    
    if not (df1["IMG_ID"] == df2["IMG_ID"]).all():
        raise ValueError("IMG_IDs do not match!")
        
    total_intersection = 0
    total_sum1 = 0
    total_sum2 = 0
    
    for idx, row in df1.iterrows():
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
        
    global_dice = (2.0 * total_intersection) / (total_sum1 + total_sum2 + 1e-6)
    return global_dice

if __name__ == '__main__':
    baseline = "final_competitive_submission.csv"
    candidates = [
        "ensemble_multiclass_w60_c3t50_area128_submission.csv",
        "ensemble_multiclass_w55_c3t50_area128_submission.csv",
        "ensemble_multiclass_w60_c3t50_area64_submission.csv"
    ]
    
    print(f"Comparing candidates against baseline: {baseline}\n")
    for cand in candidates:
        if os.path.exists(cand):
            dice = compare_submissions(baseline, cand)
            print(f"  {cand} -> Dice similarity with baseline: {dice:.6f}\n")
        else:
            print(f"  {cand} does not exist!\n")
