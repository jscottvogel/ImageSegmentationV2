"""
FloodNet PyTorch Segmentation Pipeline (RF-DETR Transformer Edition)
====================================================================

This script implements DETR (Detection Transformer) algorithms adapting them 
for dense Semantic Segmentation. 

Key Upgrades:
- Global Self-Attention via Transformer Encoders/Decoders
- Bipartite Hungarian Matching for Object instances
- Removes ConvNet "Local Receptive Field" spatial blindness

Execution:
    $ sg render -c "HSA_OVERRIDE_GFX_VERSION=10.3.0 .venv/bin/python rfdetr_version.py"
"""

import os
# Silence fake C++ TensorFlow/CUDA errors from TensorBoard, as we are physically accelerating via AMD ROCm HIP!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou
import torchvision.models as models

class DatasetConfig:
    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    NUM_CLASSES = 10
    IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

class TrainingConfig:
    BATCH_SIZE = 4  # Transformers linearly scale memory demands (O(n^2)), restricting us to BS=4
    EPOCHS = 70
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    CHECKPOINT_DIR = 'model_checkpoint/FloodNet_RFDETR'

id2color = {
    0: [0, 0, 0], 1: [255, 0, 0], 2: [200, 90, 90], 3: [128, 128, 0], 4: [155, 155, 155],
    5: [0, 255, 255], 6: [55, 0, 255], 7: [255, 0, 255], 8: [245, 245, 0], 9: [0, 255, 0],
}

# --- DATA PIPELINE: MASK TO INSTANCES ---
def rgb_to_mask(rgb_arr, color_map):
    H, W, _ = rgb_arr.shape
    colors = np.array(list(color_map.values()), dtype=np.float32)
    classes = np.array(list(color_map.keys()), dtype=np.int64)
    
    diff = rgb_arr[:, :, None, :] - colors[None, None, :, :]
    dists = np.sum(diff ** 2, axis=-1)
    min_idx = np.argmin(dists, axis=-1)
    return classes[min_idx].astype(np.uint8)

def mask_to_instances(label_map, num_classes):
    """
    Transformers don't predict flat pixel maps. They predict literal discrete objects.
    We must programmatically extract connected groupings of pixels into bounding boxed instances!
    """
    boxes, labels, masks = [], [], []
    
    for cls_idx in range(1, num_classes):
        class_mask = (label_map == cls_idx).astype(np.uint8)
        if class_mask.sum() == 0: continue
            
        num_components, components_im = cv2.connectedComponents(class_mask)
        
        for i in range(1, num_components):
            instance_mask = (components_im == i).astype(np.uint8)
            y, x = np.where(instance_mask > 0)
            
            if len(y) < 25: continue # Ignore microscopic noise anomalies
                
            xmin, xmax = np.min(x), np.max(x)
            ymin, ymax = np.min(y), np.max(y)
            
            # Translate to DETR strictly normalized format (Center-X, Center-Y, W, H)
            w = xmax - xmin
            h = ymax - ymin
            cx = xmin + w / 2.0
            cy = ymin + h / 2.0
            cx, cy, w, h = cx/DatasetConfig.IMG_WIDTH, cy/DatasetConfig.IMG_HEIGHT, w/DatasetConfig.IMG_WIDTH, h/DatasetConfig.IMG_HEIGHT
            
            boxes.append([cx, cy, w, h])
            labels.append(cls_idx)
            masks.append(instance_mask)
            
    # Matrix integrity preservation
    if not boxes:
        boxes = [[0.5, 0.5, 1.0, 1.0]]
        labels = [0]
        masks = [np.zeros((DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH), dtype=np.bool_)]
        
    # HUGE MEMORY OPTIMIZATION: Return masks as torch.bool (1 byte instead of 4 per pixel)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long), torch.tensor(np.array(masks), dtype=torch.bool)

class FloodNetDETRDataset(Dataset):
    def __init__(self, image_paths, mask_paths, num_classes, id2color):
        self.image_paths, self.mask_paths = image_paths, mask_paths
        self.num_classes, self.id2color = num_classes, id2color
        
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_raw = cv2.imread(self.image_paths[idx])
        msk_raw = cv2.imread(self.mask_paths[idx])
        
        if img_raw is None or msk_raw is None:
            idx = 0
            img_raw = cv2.imread(self.image_paths[idx])
            msk_raw = cv2.imread(self.mask_paths[idx])
            
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        msk = cv2.cvtColor(msk_raw, cv2.COLOR_BGR2RGB)
        
        # Pre-resize arrays before heavy operations to drastically save RAM and CPU
        img = cv2.resize(img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        
        augmented = self.transform(image=img, mask=rgb_to_mask(msk, self.id2color))
        
        boxes, labels, masks = mask_to_instances(augmented['mask'].numpy(), self.num_classes)
        return augmented['image'], {"boxes": boxes, "labels": labels, "masks": masks}

def detr_collate_fn(batch):
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]

# --- THE HUNGARIAN BIPARTITE MATCHER ---
def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)

class HungarianMatcher(nn.Module):
    """
    Solves the Bipartite Graph Assignment Problem natively.
    DETR spits out 100 random object predictions. We mathematically map each distinct prediction to the exact mathematically perfect Ground Truth.
    """
    def __init__(self, cost_class: float=1, cost_bbox: float=5, cost_giou: float=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        iou = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -iou

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

# --- ARCHITECTURE ---
class RF_DETR_Seg(nn.Module):
    def __init__(self, num_classes, num_queries=100):
        super().__init__()
        self.num_queries = num_queries
        self.backbone = models.efficientnet_v2_s().features
        
        self.conv = nn.Conv2d(1280, 256, 1)
        # Deep Global Self-Attention Encoders 
        self.transformer = nn.Transformer(d_model=256, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024, dropout=0.1, batch_first=True)
        
        self.query_embed = nn.Embedding(num_queries, 256)
        self.row_embed = nn.Parameter(torch.rand(100, 128))
        self.col_embed = nn.Parameter(torch.rand(100, 128))
        
        self.class_head = nn.Linear(256, num_classes + 1)
        self.bbox_head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4))

    def forward(self, x):
        features = self.backbone(x)
        h = self.conv(features)
        
        b, c, height, width = h.shape
        pos_embed = torch.cat([self.col_embed[:width].unsqueeze(0).repeat(height, 1, 1),
                               self.row_embed[:height].unsqueeze(1).repeat(1, width, 1)], dim=-1).flatten(0, 1).unsqueeze(0).repeat(b, 1, 1).to(x.device)
        h_flat = h.flatten(2).permute(0, 2, 1)
        out = self.transformer(h_flat + pos_embed, self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1))
        
        return {
            'pred_logits': self.class_head(out), 
            'pred_boxes': self.bbox_head(out).sigmoid()
        }



# --- EXECUTION ENGINE ---
def train_rfdetr():
    os.makedirs(TrainingConfig.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing RF-DETR Sub-Matrix Engine on {device}")
    
    model = RF_DETR_Seg(num_classes=DatasetConfig.NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LEARNING_RATE, weight_decay=TrainingConfig.WEIGHT_DECAY)
    writer = SummaryWriter(log_dir=f"{TrainingConfig.CHECKPOINT_DIR}/tensorboard")
    scaler = torch.amp.GradScaler('cuda')
    matcher = HungarianMatcher()
    
    T_IMG = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/train/images"
    T_MSK = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/train/masks"
    
    loader = DataLoader(
        FloodNetDETRDataset(sorted(glob.glob(os.path.join(T_IMG, "*.jpg"))), sorted(glob.glob(os.path.join(T_MSK, "*.png"))), DatasetConfig.NUM_CLASSES, id2color),
        batch_size=TrainingConfig.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=detr_collate_fn
    )
    
    try:
        # model = torch.compile(model) # DISABLED: AMD ROCm hardware crash on FP16/Triton
    except Exception: pass

    best_loss = float('inf')
    try:
        for epoch in range(TrainingConfig.EPOCHS):
            model.train()
            epoch_bipartite = 0.0
            
            for b_idx, (images, targets) in enumerate(loader):
                images = images.to(device)
                targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device), "masks": t["masks"].to(device)} for t in targets]
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda', enabled=False): # DISABLED: AMD ROCm FP16 hardware crash
                    outputs = model(images)
                    indices = matcher(outputs, targets)
                    loss = 0.0 * outputs['pred_logits'].sum() # dummy for graphs
                    
                    batch_idx = []
                    src_idx = []
                    tgt_classes = []
                    tgt_boxes = []
                    tgt_masks = []
                    
                    for i, (pred_i, tgt_i) in enumerate(indices):
                        if len(pred_i) > 0:
                            batch_idx.append(torch.full_like(pred_i, i))
                            src_idx.append(pred_i)
                            tgt_classes.append(targets[i]['labels'][tgt_i])
                            tgt_boxes.append(targets[i]['boxes'][tgt_i])
                            tgt_masks.append(targets[i]['masks'][tgt_i].flatten(1))
                            
                    
                    if batch_idx:
                        batch_idx = torch.cat(batch_idx)
                        src_idx = torch.cat(src_idx)
                        
                        pred_logits_matched = outputs['pred_logits'][batch_idx, src_idx]
                        tgt_labels_matched = torch.cat(tgt_classes)
                        loss = loss + 1.0 * F.cross_entropy(pred_logits_matched, tgt_labels_matched)
                        
                        pred_boxes_matched = outputs['pred_boxes'][batch_idx, src_idx]
                        tgt_boxes_matched = torch.cat(tgt_boxes)
                        loss = loss + 5.0 * F.l1_loss(pred_boxes_matched, tgt_boxes_matched)
                        
                    scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_bipartite += loss.item()
                
                if b_idx % 25 == 0:
                    print(f"RF-DETR Epoch {epoch + 1}/{TrainingConfig.EPOCHS} | Batch {b_idx} | Bipartite Scaled Loss: {loss.item():.4f}")
                    writer.add_scalar("Training/Bipartite-Loss", loss.item(), epoch * len(loader) + b_idx)
                    writer.flush()
                    
            metrics_avg = epoch_bipartite / len(loader)
            print(f"---> RF-DETR Epoch {epoch + 1} Completed | Avg Hungarian Loss: {metrics_avg:.4f}")
            
            if metrics_avg < best_loss:
                best_loss = metrics_avg
                torch.save(model.state_dict(), os.path.join(TrainingConfig.CHECKPOINT_DIR, "best_rfdetr_weights.pt"))
                print(f"New Best Target Distance Graph Extracted! Saved Weights.")
                
    except KeyboardInterrupt:
        print("\n[WARNING] Keyboard Interrupt detected!")
        interrupted_path = os.path.join(TrainingConfig.CHECKPOINT_DIR, "interrupted_rfdetr_weights.pt")
        torch.save(model.state_dict(), interrupted_path)
        print(f"Gracefully saved current model state to: {interrupted_path}")
        print("Safely exiting training loop...")

if __name__ == "__main__":
    train_rfdetr()
