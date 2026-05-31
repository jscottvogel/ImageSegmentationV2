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
    
    # Ensure they have same IMG_ID rows in same order
    df1 = df1.sort_values("IMG_ID").reset_index(drop=True)
    df2 = df2.sort_values("IMG_ID").reset_index(drop=True)
    
    if not (df1["IMG_ID"] == df2["IMG_ID"]).all():
        raise ValueError("IMG_IDs do not match!")
        
    total_intersection = 0
    total_sum1 = 0
    total_sum2 = 0
    
    # We can group by class if we want class-wise overlap
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
        
    # Global Dice
    global_dice = (2.0 * total_intersection) / (total_sum1 + total_sum2 + 1e-6)
    global_iou = total_intersection / (total_sum1 + total_sum2 - total_intersection + 1e-6)
    
    # Class-wise Dice
    class_dices = {}
    for c in range(10):
        class_dices[c] = (2.0 * class_intersections[c]) / (class_sums1[c] + class_sums2[c] + 1e-6)
        
    return {
        "global_dice": global_dice,
        "global_iou": global_iou,
        "class_dices": class_dices
    }

if __name__ == '__main__':
    files = {
        "DeepLabV3+": "deeplabv3plus_best_submission.csv",
        "UNet": "unet_best_submission.csv",
        "FCN": "fcn_best_submission.csv",
        "Hybrid": "hybrid_ensemble_submission.csv",
        "Meta": "meta_ensemble_submission.csv"
    }
    
    # Check if files exist
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"Error: {path} not found!")
            exit(1)
            
    print("All files found. Starting comparison...")
    
    names = list(files.keys())
    matrix_dice = np.zeros((len(names), len(names)))
    matrix_iou = np.zeros((len(names), len(names)))
    
    pairwise_results = {}
    for i in range(len(names)):
        for j in range(i, len(names)):
            if i == j:
                matrix_dice[i, j] = 1.0
                matrix_iou[i, j] = 1.0
            else:
                res = compare_submissions(files[names[i]], files[names[j]])
                matrix_dice[i, j] = res["global_dice"]
                matrix_dice[j, i] = res["global_dice"]
                matrix_iou[i, j] = res["global_iou"]
                matrix_iou[j, i] = res["global_iou"]
                pairwise_results[(names[i], names[j])] = res
                
    print("\nPairwise Dice Similarity Matrix:")
    header = "       " + "".join(f"{name:>10}" for name in names)
    print(header)
    for idx, name in enumerate(names):
        row_str = f"{name:<8}" + "".join(f"{matrix_dice[idx, j]:10.4f}" for j in range(len(names)))
        print(row_str)
        
    print("\nPairwise IoU Similarity Matrix:")
    print(header)
    for idx, name in enumerate(names):
        row_str = f"{name:<8}" + "".join(f"{matrix_iou[idx, j]:10.4f}" for j in range(len(names)))
        print(row_str)
        
    # Print class-wise details for Hybrid vs Meta
    if ("Hybrid", "Meta") in pairwise_results:
        print("\nClass-wise Dice Similarity (Hybrid vs Meta):")
        for c, score in pairwise_results[("Hybrid", "Meta")]["class_dices"].items():
            print(f"Class {c}: {score:.4f}")
            
    # Print class-wise details for Meta vs DeepLabV3+
    if ("DeepLabV3+", "Meta") in pairwise_results:
        print("\nClass-wise Dice Similarity (DeepLabV3+ vs Meta):")
        for c, score in pairwise_results[("DeepLabV3+", "Meta")]["class_dices"].items():
            print(f"Class {c}: {score:.4f}")
