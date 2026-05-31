import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import glob
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
from scipy.ndimage import distance_transform_edt

from synergistic_model import FloodNetSynergisticNet
from optimized_pytorch_version import DatasetConfig, id2color, rgb_to_mask
from scratch.eval_advanced_tta import evaluate_config

def main():
    print("Evaluating blends without TTA...")
    
    blends = [
        ("No TTA, No Blending", 1.0, 0.0, 0.0, 0.0),
        ("No TTA, 0.8 Main + 0.1 UNet + 0.1 DeepLab", 0.8, 0.1, 0.1, 0.0),
        ("No TTA, 0.7 Main + 0.15 UNet + 0.15 DeepLab", 0.7, 0.15, 0.15, 0.0),
        ("No TTA, 0.6 Main + 0.2 UNet + 0.2 DeepLab", 0.6, 0.2, 0.2, 0.0),
        ("No TTA, 0.5 Main + 0.2 UNet + 0.2 DeepLab + 0.1 FCN", 0.5, 0.2, 0.2, 0.1),
        ("No TTA, 0.4 Main + 0.3 UNet + 0.3 DeepLab", 0.4, 0.3, 0.3, 0.0)
    ]
    
    for desc, wm, wu, wd, wf in blends:
        score = evaluate_config(wm, wu, wd, wf, ["none"], 0.0, 100)
        print(f"Config: {desc:50s} | Score: {score:.5f}")

if __name__ == '__main__':
    main()
