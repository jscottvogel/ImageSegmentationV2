import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import glob
import cv2
cv2.setNumThreads(0)

import logging
import traceback

logging.basicConfig(
    filename='unet_trace.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("UNet Diagnostics Tracer Initialized")

# 1. Borrow pristine multiscale utilities and datasets inherently built
from optimized_pytorch_version import (
    FloodNetPyTorchDataset, DatasetConfig, TrainingConfig, 
    id2color, soft_dice_loss, wce_standard, ftl, active_contour_loss, class_weights
)
from unet_version import StandardUNet

def train_unet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Standard PyTorch UNet on {device}")
    
    # Target directory structure safely
    os.makedirs("model_checkpoint/FloodNet_UNet", exist_ok=True)
    
    # 2. Instantiate Base Model
    model = StandardUNet(num_classes=DatasetConfig.NUM_CLASSES).to(device)
    # Disabled torch.compile() to prevent 'HIP error: invalid device function' 
    # hardware crashes on AMD GPUs during backward/optimizer steps.
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LEARNING_RATE, weight_decay=TrainingConfig.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    
    # 3. Mount Native FloodNet Datasets robustly
    T_IMG = DatasetConfig.TRAIN_IMG_DIR
    T_MSK = DatasetConfig.TRAIN_MSK_DIR
    
    dataset = FloodNetPyTorchDataset(sorted(glob.glob(os.path.join(T_IMG, "*.jpg"))), sorted(glob.glob(os.path.join(T_MSK, "*.png"))), DatasetConfig.NUM_CLASSES, id2color)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True) # Reduced BS to 4 to prevent VRAM OOM
    
    best_loss = float('inf')
    for epoch in range(TrainingConfig.EPOCHS):
        model.train()
        
        t_hard_i = torch.zeros(DatasetConfig.NUM_CLASSES, device=device)
        t_hard_u = torch.zeros(DatasetConfig.NUM_CLASSES, device=device)
        epoch_loss = 0.0
        
        for b_idx, (images, targets) in enumerate(loader):
            try:
                logging.debug(f"--- Epoch {epoch+1} Batch {b_idx} ---")
                images = images.to(device)
                labels = targets['main_output'].to(device)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Executing natively in FP32 to prevent AMD / ROCm FP16 kernel crashes
                logging.debug("Starting Forward Pass")
                preds = model(images)['main_output']
                
                logging.debug("Calculating Soft Dice Loss")
                d_loss, _, _ = soft_dice_loss(preds, labels)
                logging.debug("Calculating WCE Loss")
                wce_loss = wce_standard(preds, labels)
                logging.debug("Calculating FTL Loss")
                f_loss = ftl(preds, labels)
                logging.debug("Calculating Active Contour Loss")
                ac_loss = active_contour_loss(preds, labels)
                
                # Combine robust composite losses matching DeepLab Phase 1
                loss = (0.2 * wce_loss) + (0.3 * f_loss) + (0.4 * d_loss) + (0.1 * ac_loss)
                
                logging.debug("Backward Pass starts")
                loss.backward()
                
                logging.debug("Optimizer step starts")
                optimizer.step()
                
                epoch_loss += loss.item()
    
                if torch.cuda.is_available():
                    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    logging.debug(f"Batch {b_idx} success. Active VRAM: {vram_mb:.1f} MB")
                
                with torch.no_grad():
                    pred_labels = torch.argmax(preds, dim=1)
                    import torch.nn.functional as F
                    hard_pred = F.one_hot(pred_labels, DatasetConfig.NUM_CLASSES).permute(0,3,1,2).float()
                    hard_true = F.one_hot(labels, DatasetConfig.NUM_CLASSES).permute(0,3,1,2).float()
                    t_hard_i += torch.sum(hard_pred * hard_true, (0,2,3))
                    t_hard_u += torch.sum(hard_pred + hard_true, (0,2,3))
                
                if b_idx % 25 == 0:
                    print(f"UNet Epoch {epoch + 1}/{TrainingConfig.EPOCHS} | Batch {b_idx} | Phase-1 Composite Loss: {loss.item():.4f}")
            except Exception as e:
                logging.error(f"CRITICAL CRASH ON EPOCH {epoch+1} BATCH {b_idx}")
                logging.error(traceback.format_exc())
                print(f"CRITICAL ERROR CAUGHT: See unet_trace.log for detailed traceback.")
                raise e
                
        scheduler.step()
        
        # Calculate exactly like DeepLab to get True Mode Accuracy
        cw = torch.clamp(class_weights.clone().detach().to(device), min=0.0)
        cw = cw / (torch.sum(cw) + 1e-6)
        true_hard_dice = torch.sum(((2. * t_hard_i + 1.0) / (t_hard_u + 1.0)) * cw).item()
        
        metrics_avg = epoch_loss / len(loader)
        print(f"---> UNet Epoch {epoch + 1} Completed | True Hard Dice Accuracy: {true_hard_dice:.4f} | Avg Loss: {metrics_avg:.4f}")
        
        # Save exact UNet geometry format
        if metrics_avg < best_loss:
            best_loss = metrics_avg
            torch.save(model.state_dict(), "model_checkpoint/FloodNet_UNet/best_unet_weights.pt")
            print("New Best UNet Checkpoint Saved!")

if __name__ == '__main__':
    train_unet()
