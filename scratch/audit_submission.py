import os
import pandas as pd
import numpy as np

def audit_submission(file_path):
    print(f"Auditing: {file_path}")
    
    if not os.path.exists(file_path):
        print("FAIL: File does not exist!")
        return False
        
    df = pd.read_csv(file_path)
    
    # Check shape
    if len(df) != 5000:
        print(f"FAIL: Expected exactly 5000 rows, got {len(df)}")
        return False
        
    # Check headers
    if list(df.columns) != ["IMG_ID", "EncodedString"]:
        print(f"FAIL: Headers mismatch, got {df.columns}")
        return False
        
    # Check row format: image_id_class
    img_ids = df["IMG_ID"].tolist()
    for idx, iid in enumerate(img_ids):
        parts = iid.split('_')
        if len(parts) != 2 or not parts[1].isdigit() or len(parts[1]) != 2:
            print(f"FAIL: Invalid IMG_ID format at row {idx+1}: {iid}")
            return False
            
    # Check EncodedString content
    non_empty = 0
    zero_indexed_count = 0
    for idx, row in df.iterrows():
        enc = row["EncodedString"]
        if pd.isna(enc) or str(enc).strip() == "":
            continue
        non_empty += 1
        
        # Verify run-length values
        try:
            runs = [int(x) for x in str(enc).split()]
        except ValueError:
            print(f"FAIL: EncodedString contains non-integer value at row {idx+1}")
            return False
            
        if len(runs) % 2 != 0:
            print(f"FAIL: Odd number of RLE values at row {idx+1}")
            return False
            
        # Verify 1-based indexing
        starts = runs[0::2]
        lengths = runs[1::2]
        
        if any(s < 1 for s in starts):
            zero_indexed_count += 1
            print(f"WARNING: Zero or negative index found at row {idx+1}: {starts[0]}")
            
        if any(l <= 0 for l in lengths):
            print(f"FAIL: Negative or zero length found at row {idx+1}")
            return False
            
    print(f"SUCCESS: {non_empty} non-empty RLE fields verified.")
    if zero_indexed_count > 0:
        print(f"WARNING: {zero_indexed_count} rows contain 0-based start indices.")
    else:
        print("Index Validation: Verified 1-based start indices successfully.")
    print("-" * 50)
    return True

if __name__ == "__main__":
    files = [
        "synergistic_optimized_t95_submission.csv",
        "synergistic_optimized_t99_submission.csv",
        "synergistic_optimized_t995_submission.csv"
    ]
    for f in files:
        audit_submission(f)
