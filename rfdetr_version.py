import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

# Re-use the exact same rock-solid Dataset and Loss functions that successfully trained DeepLab!
from optimized_pytorch_version import (
    FloodNetPyTorchDataset, id2color, DatasetConfig,
    soft_dice_loss, wce_standard
)

class TrainingConfig:
    BATCH_SIZE = 4  # Transformers use O(n^2) memory in self-attention, requiring a small batch size
    EPOCHS = 70
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    CHECKPOINT_DIR = 'model_checkpoint/FloodNet_Transformer'

class DenseTransformerSegmentation(nn.Module):
    """
    A Global Self-Attention Segmentation Network.
    Unlike CNNs which are blind to anything outside their small kernel window, this Transformer
    flattens the entire image into tokens and allows every pixel to calculate its relationship
    with every other pixel on the entire map simultaneously. 
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # 1. Local Feature Extractor
        self.backbone = models.efficientnet_v2_s().features
        self.conv = nn.Conv2d(1280, 256, 1)
        
        # 2. Global Self-Attention Encoders
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 3. Positional Embeddings (So the transformer knows where pixels are physically located)
        self.row_embed = nn.Parameter(torch.rand(100, 128))
        self.col_embed = nn.Parameter(torch.rand(100, 128))
        
        # 4. Dense Decoder Head (Upsamples back to full drone resolution)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),
            
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False),
            
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        # Local features
        features = self.backbone(x)
        h = self.conv(features)
        
        b, c, height, width = h.shape
        
        # Inject structural positioning
        pos_embed = torch.cat([
            self.col_embed[:width].unsqueeze(0).repeat(height, 1, 1),
            self.row_embed[:height].unsqueeze(1).repeat(1, width, 1)
        ], dim=-1).flatten(0, 1).unsqueeze(0).repeat(b, 1, 1).to(x.device)
        
        h_flat = h.flatten(2).permute(0, 2, 1)
        
        # GLOBAL ATTENTION (Every pixel talks to every other pixel)
        memory = self.transformer_encoder(h_flat + pos_embed)
        
        # Re-fold back into spatial map
        memory_spatial = memory.permute(0, 2, 1).view(b, c, height, width)
        
        # Decode to Dense Output Matrix
        out_mask = self.decoder(memory_spatial)
        
        # Final rigorous snap to original physical dimensions
        out_mask = F.interpolate(out_mask, size=(x.shape[2], x.shape[3]), mode='bicubic', align_corners=False)
        
        return {'main_output': out_mask}

def train_transformer():
    os.makedirs(TrainingConfig.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Dense Transformer Engine on {device}")
    
    model = DenseTransformerSegmentation(num_classes=DatasetConfig.NUM_CLASSES).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LEARNING_RATE, weight_decay=TrainingConfig.WEIGHT_DECAY)
    writer = SummaryWriter(log_dir=f"{TrainingConfig.CHECKPOINT_DIR}/tensorboard")
    scaler = torch.amp.GradScaler('cuda')
    
    T_IMG = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/train/images"
    T_MSK = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/train/masks"
    
    loader = DataLoader(
        FloodNetPyTorchDataset(sorted(glob.glob(os.path.join(T_IMG, "*.jpg"))), sorted(glob.glob(os.path.join(T_MSK, "*.png"))), DatasetConfig.NUM_CLASSES, id2color),
        batch_size=TrainingConfig.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    
    best_loss = float('inf')
    
    try:
        for epoch in range(TrainingConfig.EPOCHS):
            model.train()
            epoch_loss = 0.0
            
            for b_idx, (images, mask_dict) in enumerate(loader):
                images = images.to(device)
                masks = mask_dict['main_output'].to(device)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Disabled autocast due to ROCm FP16 architecture crashes on current drivers
                with torch.amp.autocast('cuda', enabled=False):
                    pred_logits = model(images)['main_output']
                    
                    ce_loss = wce_standard(pred_logits, masks, DatasetConfig.NUM_CLASSES)
                    dice_loss_val, _, _ = soft_dice_loss(pred_logits, masks, DatasetConfig.NUM_CLASSES)
                    
                    loss = ce_loss + (dice_loss_val * 2.0)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                
                if b_idx % 25 == 0:
                    print(f"Transformer Epoch {epoch + 1}/{TrainingConfig.EPOCHS} | Batch {b_idx} | CE: {ce_loss.item():.4f} | Dice: {dice_loss_val.item():.4f} | Total Loss: {loss.item():.4f}")
                    writer.add_scalar("Training/Loss", loss.item(), epoch * len(loader) + b_idx)
                    writer.flush()
                    
            metrics_avg = epoch_loss / len(loader)
            print(f"---> Transformer Epoch {epoch + 1} Completed | Avg Loss: {metrics_avg:.4f}")
            
            if metrics_avg < best_loss:
                best_loss = metrics_avg
                torch.save(model.state_dict(), os.path.join(TrainingConfig.CHECKPOINT_DIR, "best_transformer_weights.pt"))
                print(f"New Best Global Attention Extracted! Saved Transformer Weights.")
                
    except KeyboardInterrupt:
        print("\n[WARNING] Keyboard Interrupt detected!")
        interrupted_path = os.path.join(TrainingConfig.CHECKPOINT_DIR, "interrupted_transformer_weights.pt")
        torch.save(model.state_dict(), interrupted_path)
        print(f"Gracefully saved current model state to: {interrupted_path}")

if __name__ == "__main__":
    train_transformer()
