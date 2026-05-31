import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"

import time
import torch
import glob
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from optimized_pytorch_version import (
    FloodNetPyTorchDataset, DatasetConfig, TrainingConfig, 
    id2color, CustomDeepLabV3Plus
)

def benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    model = CustomDeepLabV3Plus(num_classes=10).to(device)
    model.train()
    
    tr_img = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))[:100]
    tr_msk = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))[:100]
    
    dataset = FloodNetPyTorchDataset(tr_img, tr_msk, DatasetConfig.NUM_CLASSES, id2color)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=2)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    t_start = time.time()
    t_data_total = 0.0
    t_forward_total = 0.0
    t_backward_total = 0.0
    t_step_total = 0.0
    
    t_loader_start = time.time()
    for b_idx, (images, targets) in enumerate(loader):
        t_data = time.time() - t_loader_start
        t_data_total += t_data
        
        t_f_start = time.time()
        images = images.to(device, non_blocking=True)
        labels = targets['main_output'].to(device, non_blocking=True)
        preds = model(images)['main_output']
        
        # Loss computation (CrossEntropy for simple benchmark)
        loss = F.cross_entropy(preds, labels, ignore_index=255)
        t_forward = time.time() - t_f_start
        t_forward_total += t_forward
        
        t_b_start = time.time()
        loss.backward()
        t_backward = time.time() - t_b_start
        t_backward_total += t_backward
        
        t_s_start = time.time()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        t_step = time.time() - t_s_start
        t_step_total += t_step
        
        print(f"Batch {b_idx} | Data: {t_data:.4f}s | Forward: {t_forward:.4f}s | Backward: {t_backward:.4f}s | Step: {t_step:.4f}s")
        
        if b_idx >= 10:
            break
        t_loader_start = time.time()
        
    t_total = time.time() - t_start
    print(f"\nAverage Times over {b_idx + 1} batches:")
    print(f"Data Loading: {t_data_total / (b_idx + 1):.4f}s")
    print(f"Forward Pass: {t_forward_total / (b_idx + 1):.4f}s")
    print(f"Backward Pass: {t_backward_total / (b_idx + 1):.4f}s")
    print(f"Optimizer Step: {t_step_total / (b_idx + 1):.4f}s")
    print(f"Total Time: {t_total:.4f}s")

if __name__ == '__main__':
    benchmark()
