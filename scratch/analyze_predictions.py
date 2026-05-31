import os
import pandas as pd
import numpy as np

# Class names
id2labels = {
    0: 'Background/waterbody',
    1: 'Building Flooded',
    2: 'Building Non-Flooded',
    3: 'Road Flooded',
    4: 'Road Non-Flooded',
    5: 'Water',
    6: 'Tree',
    7: 'Vehicle',
    8: 'Pool',
    9: 'Grass',
}

submission_files = [
    ("DeepLabV3+", "deeplabv3plus_best_submission.csv"),
    ("UNet", "unet_best_submission.csv"),
    ("FCN", "fcn_best_submission.csv"),
    ("Hybrid Ensemble", "hybrid_ensemble_submission.csv"),
    ("Meta Ensemble", "meta_ensemble_submission.csv"),
    ("Meta Ensemble (TTA)", "meta_ensemble_submission_tta.csv")
]

def analyze_file(file_path):
    if not os.path.exists(file_path):
        return None
        
    df = pd.read_csv(file_path)
    class_pixels = {c: 0 for c in range(10)}
    
    for _, row in df.iterrows():
        img_id = str(row["IMG_ID"])
        class_id = int(img_id.split('_')[-1])
        
        rle = row["EncodedString"]
        if pd.isna(rle) or str(rle).strip().lower() in ['nan', '']:
            continue
            
        parts = [int(x) for x in str(rle).strip().split()]
        lengths = parts[1::2]
        class_pixels[class_id] += sum(lengths)
        
    total_pixels = sum(class_pixels.values())
    if total_pixels == 0:
        return {c: 0.0 for c in range(10)}
        
    # Return percentages of total predicted pixels
    return {c: (class_pixels[c] / total_pixels) * 100 for c in range(10)}

def main():
    results = {}
    for name, f in submission_files:
        results[name] = analyze_file(f)
        
    print("| Class ID | Class Name | " + " | ".join(results.keys()) + " |")
    print("|---" * (len(results) + 2) + "|")
    for c in range(10):
        row_str = f"| {c} | {id2labels[c]} |"
        for name in results.keys():
            val = results[name][c] if results[name] is not None else 0.0
            row_str += f" {val:.2f}% |"
        print(row_str)

if __name__ == '__main__':
    main()
