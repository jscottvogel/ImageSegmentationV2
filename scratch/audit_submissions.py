import os
import pandas as pd
import numpy as np

def audit_submission(path, name):
    print(f"\n================ AUDITING {name} ================")
    print(f"File path: {path}")
    
    if not os.path.exists(path):
        print(f"ERROR: File {path} does not exist!")
        return False
        
    df = pd.read_csv(path)
    
    # 1. Row count check
    expected_rows = 5000
    if len(df) != expected_rows:
        print(f"ERROR: Expected {expected_rows} rows, but got {len(df)}!")
        return False
    else:
        print(f"SUCCESS: Row count is exactly {expected_rows}.")
        
    # 2. Columns check
    expected_cols = ["IMG_ID", "EncodedString"]
    if list(df.columns) != expected_cols:
        print(f"ERROR: Expected columns {expected_cols}, but got {list(df.columns)}!")
        return False
    else:
        print("SUCCESS: Column names are correct.")
        
    # 3. NaNs or Nulls check in IMG_ID
    if df["IMG_ID"].isnull().any():
        print("ERROR: Found null values in IMG_ID!")
        return False
        
    # 4. Check Class 0 prediction stats
    df0 = df[df['IMG_ID'].str.endswith('_00')]
    non_empty = df0[df0['EncodedString'].notna() & (df0['EncodedString'] != '')]
    print(f"Total images with Class 0 (Background) predictions: {len(non_empty)}")
    
    # 5. Check RLE index sanity
    # Sample a few non-empty RLEs and check that indices are 1-based and increasing
    non_empty_rles = df[df['EncodedString'].notna() & (df['EncodedString'] != '')]['EncodedString'].head(10)
    for idx, rle in enumerate(non_empty_rles):
        parts = [int(x) for x in rle.split()]
        if len(parts) % 2 != 0:
            print(f"ERROR: RLE string length is not even!")
            return False
        starts = parts[::2]
        lengths = parts[1::2]
        
        # Check 1-based start
        if any(s <= 0 for s in starts):
            print(f"ERROR: Found non-positive start index in RLE: {starts}")
            return False
            
        # Check sorted starts (must be strictly increasing)
        for i in range(1, len(starts)):
            if starts[i] <= starts[i-1] + lengths[i-1]:
                print(f"ERROR: Overlapping or non-monotonic runs in RLE!")
                return False
                
    print("SUCCESS: RLE encoding passed all index sanity checks.")
    return True

def main():
    paths = {
        'Competitive Baseline': 'final_competitive_submission.csv',
        'Synergistic Calibrated T50': 'synergistic_calibrated_t50_submission.csv',
        'Synergistic Calibrated T90': 'synergistic_calibrated_t90_submission.csv',
        'Synergistic Calibrated T95': 'synergistic_calibrated_t95_submission.csv',
        'Synergistic Calibrated T99': 'synergistic_calibrated_t99_submission.csv',
        'Synergistic Calibrated T100': 'synergistic_calibrated_t100_submission.csv',
        'Synergistic Neighbor Filled T90': 'synergistic_neighbor_filled_t90_submission.csv',
        'Synergistic Neighbor Filled T95': 'synergistic_neighbor_filled_t95_submission.csv',
        'Synergistic Neighbor Filled T99': 'synergistic_neighbor_filled_t99_submission.csv',
        'Synergistic Optimized T90': 'synergistic_optimized_t90_submission.csv',
        'Synergistic Optimized T95': 'synergistic_optimized_t95_submission.csv',
        'Synergistic Optimized T99': 'synergistic_optimized_t99_submission.csv',
        'Blended Ensemble Submission': 'blended_ensemble_submission.csv',
        'Synergistic Pure TTA T95 Area 25': 'synergistic_pure_tta_t95_area25_submission.csv',
        'Synergistic Pure TTA T90 Area 8': 'synergistic_pure_tta_t90_area8_submission.csv',
        'Hybrid Pure TTA T95 Area 25': 'hybrid_pure_tta_t95_area25_submission.csv',
        'Synergistic TTA Suppress C0 C3': 'synergistic_tta_suppress_c0_c3_submission.csv',
        'Hybrid TTA Suppress C0 C3': 'hybrid_tta_suppress_c0_c3_submission.csv',
        'Hybrid TTA Suppress C0 C3 C1': 'hybrid_tta_suppress_c0_c3_c1_submission.csv',
        'Ensemble w50 t95 C3 C1 Area64 (No-TTA)': 'ensemble_w50_t95_c3t50_c1t50_area64_submission.csv',
        'Ensemble w50 t100 C3 C1 Area64 (No-TTA)': 'ensemble_w50_t100_c3t50_c1t50_area64_submission.csv',
        'Ensemble w50 t100 C3 C1 Area128 (No-TTA)': 'ensemble_w50_t100_c3t50_c1t50_area128_submission.csv',
        'Ensemble w50 t90 C3 C1 Area128 (No-TTA)': 'ensemble_w50_t90_c3t50_c1t50_area128_submission.csv',
        'Ensemble w50 t95 C3 C1 Area128 (No-TTA)': 'ensemble_w50_t95_c3t50_c1t50_area128_submission.csv',
        'Final Kaggle Submission': 'final_kaggle_submission.csv'
    }
    
    for name, path in paths.items():
        if os.path.exists(path):
            audit_submission(path, name)
        else:
            print(f"\n[INFO] {name} ({path}) not found yet. Skipping audit for now.")

if __name__ == '__main__':
    main()
