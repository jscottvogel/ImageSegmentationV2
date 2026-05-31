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
    
    class_intersections = {c: 0 for c in range(10)}
    class_sums1 = {c: 0 for c in range(10)}
    class_sums2 = {c: 0 for c in range(10)}
    
    for idx, row in tqdm(df1.iterrows(), total=len(df1), desc=f"Comparing {os.path.basename(file1)} vs {os.path.basename(file2)}"):
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
    class_dices = {}
    for c in range(10):
        class_dices[c] = (2.0 * class_intersections[c]) / (class_sums1[c] + class_sums2[c] + 1e-6)
        
    return {
        "global_dice": global_dice,
        "class_dices": class_dices,
        "sum1": total_sum1,
        "sum2": total_sum2
    }

if __name__ == '__main__':
    files = {
        "Synergistic": "synergistic_submission.csv",
        "Meta_TTA": "meta_ensemble_submission_tta.csv",
        "Final_Comp": "final_competitive_submission.csv",
        "Meta": "meta_ensemble_submission.csv"
    }
    
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"Error: {path} not found!")
            exit(1)
            
    print("Files verified. Running comparisons...")
    
    # We will compare everything to Final_Comp (which likely achieved 89.323%)
    ref = "Final_Comp"
    for name in files.keys():
        if name == ref: continue
        res = compare_submissions(files[ref], files[name])
        print(f"\n{ref} vs {name}:")
        print(f"  Global Dice Similarity: {res['global_dice']:.4f}")
        print(f"  Pixel count in {ref}: {res['sum1']}")
        print(f"  Pixel count in {name}: {res['sum2']}")
        print("  Class-wise Dice:")
        for c in range(10):
            print(f"    Class {c}: {res['class_dices'][c]:.4f}")
