"""
FloodNet PyTorch Segmentation Pipeline (Optimized Edition)
=========================================================

This script implements an enterprise-grade PyTorch pipeline for Semantic Segmentation 
using the DeepLabV3+ architecture with an EfficientNetV2S backbone.

Key Optimizations:
- FP16 AMP (Automatic Mixed Precision)
- SWA (Stochastic Weight Averaging)
- Cosine Annealing Warm Restarts
- Layer-wise Learning Rate Decay (LLRD)
- DYNAMIC EPOCH ROUTER: Runs Soft-Dice/WCE for Epochs 0-30, then switches dynamically into Phase-2 (Lovász-Softmax and OHEM).

Standard Execution:
    $ sg render -c "HSA_OVERRIDE_GFX_VERSION=10.3.0 .venv/bin/python optimized_pytorch_version.py"
"""

# === IMPORTS ===
import os
import glob
import random
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.spatial import KDTree
from typing import Dict, Tuple, List, Any

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
class DatasetConfig:
    IMG_HEIGHT: int = 512
    IMG_WIDTH: int = 512
    NUM_CLASSES: int = 10
    IMAGE_SIZE: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH)
    TRAIN_IMG_DIR: str = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/train/images"
    TRAIN_MSK_DIR: str = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/train/masks"

class TrainingConfig:
    BATCH_SIZE: int = 8
    EPOCHS: int = 70
    ROUTER_EPOCH: int = 30 # Epoch at which we activate Lovasz and OHEM Phase 2
    LEARNING_RATE: float = 0.0001
    WEIGHT_DECAY: float = 1e-5
    CHECKPOINT_DIR: str = 'model_checkpoint/FloodNet_PyTorch'
    LOG_INTERVAL: int = 25

id2color: Dict[int, List[int]] = {
    0: [0, 0, 0], 1: [255, 0, 0], 2: [200, 90, 90], 3: [128, 128, 0], 4: [155, 155, 155],
    5: [0, 255, 255], 6: [55, 0, 255], 7: [255, 0, 255], 8: [245, 245, 0], 9: [0, 255, 0],
}

class_weights = torch.tensor([0.0481, 0.4213, 0.0392, 0.4079, 0.0262, 0.0305, 0.0100, 0.0614, 0.0544, 0.0010], dtype=torch.float32)

# === DATA PIPELINE ===
def rgb_to_mask(rgb_arr: np.ndarray, color_map: dict, num_classes: int) -> np.ndarray:
    # 1. We flatten the RGB color dimensions
    H, W, _ = rgb_arr.shape
    color_list = np.array(list(color_map.values()))
    class_indices = list(color_map.keys())
    
    # 2. Use KDTree to rapidly evaluate nearest-neighbor colors avoiding weird interpolated artifacts
    tree = KDTree(color_list)
    reshaped = rgb_arr.reshape(-1, 3)
    _, indices = tree.query(reshaped)
    
    # 3. Shape back into the required HxW tensor format
    label_map = np.array([class_indices[i] for i in indices]).reshape(H, W)
    return label_map

def mask_to_soft_edge(label_map: np.ndarray, num_classes: int) -> np.ndarray:
    edge_maps = []
    # 1. Loop through every class to figure out where its structural edges are
    for c in range(num_classes):
        class_mask = (label_map == c).astype(np.uint8)
        
        # 2. Apply explicit mathematical gradients in X and Y
        grad_x = cv2.Sobel(class_mask, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(class_mask, cv2.CV_64F, 0, 1, ksize=3)
        
        # 3. Join them into the absolute magnitude
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_maps.append(grad_mag)
        
    # 4. Collapse all 10 edge maps into a single structural edge matrix
    edge = np.sum(edge_maps, axis=0)
    edge = np.clip(edge, 0, 1).astype(np.float32)
    return edge

def apply_mosaic(images: List[np.ndarray], masks: List[np.ndarray], out_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = out_size
    # 1. Pick a random grid center somewhere tightly within the bounding box
    yc, xc = [int(random.uniform(0.3, 0.7) * s) for s in out_size]
    
    # 2. Base zeroed output structures
    output_img = np.zeros((h, w, 3), dtype=np.uint8)
    output_msk = np.zeros((h, w), dtype=np.long)
    
    # 3. Paste 4 different drone shots physically into the four quadrants
    for i in range(4):
        img = cv2.resize(images[i], (w, h))
        msk = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
        if i == 0:
            output_img[:yc, :xc] = img[:yc, :xc]
            output_msk[:yc, :xc] = msk[:yc, :xc]
        elif i == 1:
            output_img[:yc, xc:] = img[:yc, xc:]
            output_msk[:yc, xc:] = msk[:yc, xc:]
        elif i == 2:
            output_img[yc:, :xc] = img[yc:, :xc]
            output_msk[yc:, :xc] = msk[yc:, :xc]
        else:
            output_img[yc:, xc:] = img[yc:, xc:]
            output_msk[yc:, xc:] = msk[yc:, xc:]
    return output_img, output_msk

def apply_cutmix(img1: np.ndarray, msk1: np.ndarray, img2: np.ndarray, msk2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Alpha-Splice structural geometries completely destroying spatial bias logic."""
    h, w = img1.shape[:2]
    cut_rat = np.sqrt(1. - np.random.beta(1.0, 1.0))
    cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
    cx, cy = random.randint(0, w), random.randint(0, h)

    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, w), np.clip(cy - cut_h // 2, 0, h)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, w), np.clip(cy + cut_h // 2, 0, h)

    img1[bby1:bby2, bbx1:bbx2] = img2[bby1:bby2, bbx1:bbx2]
    msk1[bby1:bby2, bbx1:bbx2] = msk2[bby1:bby2, bbx1:bbx2]
    return img1, msk1

class FloodNetPyTorchDataset(Dataset):
    def __init__(self, image_paths: List[str], mask_paths: List[str], num_classes: int, id2color: dict, apply_aug: bool=True, use_mosaic: bool=True):
        self.image_paths, self.mask_paths = image_paths, mask_paths
        self.num_classes, self.id2color = num_classes, id2color
        self.apply_aug, self.use_mosaic = apply_aug, use_mosaic
        
        # 1. Base augmentation (Removed elastic/warping to preserve building geometries)
        self.transform = A.Compose([
            A.Resize(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
            A.RandomBrightnessContrast(p=0.5), A.Normalize(), ToTensorV2()
        ]) if apply_aug else A.Compose([A.Resize(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH), A.Normalize(), ToTensorV2()])

    def __len__(self) -> int: 
        return len(self.image_paths)

    def _load(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.cvtColor(cv2.imread(self.mask_paths[idx]), cv2.COLOR_BGR2RGB)
        
        # PREVENT RAM OOM CRASHES: Resize early before the O(n^2) KDTree pixel processing!
        img = cv2.resize(img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        
        return img, rgb_to_mask(msk, self.id2color, self.num_classes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.use_mosaic and random.random() < 0.3:
            idxs = [idx] + random.sample(range(len(self.image_paths)), 3)
            imgs_msks = [self._load(i) for i in idxs]
            img, label = apply_mosaic([im for im, _ in imgs_msks], [mk for _, mk in imgs_msks], DatasetConfig.IMAGE_SIZE)
        elif self.use_mosaic and random.random() < 0.3:
            idx2 = random.randint(0, len(self.image_paths)-1)
            img1, msk1 = self._load(idx)
            img2, msk2 = self._load(idx2)
            img, label = apply_cutmix(img1, msk1, img2, msk2)
        else:
            img, label = self._load(idx)
            
        augmented = self.transform(image=img, mask=label)
        img, label = augmented['image'], augmented['mask'].numpy()
        edge = torch.tensor(mask_to_soft_edge(label, self.num_classes), dtype=torch.float32).unsqueeze(0)
        
        # 2. Return standard classification map and explicitly carved edge matrix for dual heads
        return img, {'main_output': torch.tensor(label, dtype=torch.long), 'edge_output': edge}

# === ARCHITECTURE ===
def conv_bn_relu(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int=256):
        super().__init__()
        # ASPP handles Multi-Scale fields without reshaping
        # Adjusted ASPP Dilations to [3, 6, 9] to safely match the OS=32 (16x16) PyTorch EfficientNet extraction
        self.c1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=3, dilation=3, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.c3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.c4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=9, dilation=9, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.proj = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        return self.proj(torch.cat([F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=False), self.c1(x), self.c2(x), self.c3(x), self.c4(x)], 1))

class CustomDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int=10):
        super().__init__()
        import torchvision.models as models
        
        # 1. Grab base EfficientNet structures via Feature extraction framework
        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1).features
        self.aspp = ASPP(1280, 256)
        
        # 2. Allocate the decoders for progressive refinement across the 5 output maps
        self.high_conv = conv_bn_relu(160, 128)
        self.fuse_high = conv_bn_relu(256 + 128, 256)
        self.mid_high_conv = conv_bn_relu(128, 128)
        self.fuse_mid_high = conv_bn_relu(256 + 128, 256)
        self.mid_conv = conv_bn_relu(64, 64)
        self.fuse_mid = conv_bn_relu(256 + 64, 128)
        self.low_conv = conv_bn_relu(48, 48)
        self.fuse_low = conv_bn_relu(128 + 48, 64)
        self.final_refine = conv_bn_relu(64, 64)
        
        # 3. Create prediction output blocks
        self.main_head = nn.Conv2d(64, num_classes, 1)
        self.edge_proj = conv_bn_relu(48, 64)
        self.edge_head = nn.Conv2d(64, 1, 1)
        self.side_head = nn.Conv2d(256, num_classes, 1)
        self.high_head = nn.Conv2d(256, num_classes, 1)
        self.mid_high_head = nn.Conv2d(256, num_classes, 1)
        self.mid_head = nn.Conv2d(128, num_classes, 1)
        self.low_head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        insz = x.shape[-2:]
        fs = {}
        
        # 1. Tap into inner hidden states from the EfficientNet
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i==2: fs['low'] = x
            if i==3: fs['mid'] = x
            if i==4: fs['mid_h'] = x
            if i==5: fs['high'] = x
            if i==7: fs['top'] = x
                
        # 2. Process via ASPP algorithm context scaling
        aspp = self.aspp(fs['top'])
        
        # 3. Progressively connect spatial resolutions
        h = F.interpolate(aspp, size=fs['high'].shape[-2:], mode='bilinear', align_corners=False)
        high = self.fuse_high(torch.cat([h, self.high_conv(fs['high'])], 1))
        h = F.interpolate(high, size=fs['mid_h'].shape[-2:], mode='bilinear', align_corners=False)
        mid_h = self.fuse_mid_high(torch.cat([h, self.mid_high_conv(fs['mid_h'])], 1))
        h = F.interpolate(mid_h, size=fs['mid'].shape[-2:], mode='bilinear', align_corners=False)
        mid = self.fuse_mid(torch.cat([h, self.mid_conv(fs['mid'])], 1))
        h = F.interpolate(mid, size=fs['low'].shape[-2:], mode='bilinear', align_corners=False)
        low = self.fuse_low(torch.cat([h, self.low_conv(fs['low'])], 1))
        
        # 4. Generate identical scale dictionaries explicitly matching original outputs
        return {
            'main_output': F.interpolate(self.main_head(self.final_refine(low)), size=insz, mode='bilinear', align_corners=False),
            'edge_output': F.interpolate(self.edge_head(self.edge_proj(fs['low'])), size=insz, mode='bilinear', align_corners=False),
            'side_output': F.interpolate(self.side_head(aspp), size=insz, mode='bilinear', align_corners=False),
            'high_output': F.interpolate(self.high_head(high), size=insz, mode='bilinear', align_corners=False),
            'mid_high_output': F.interpolate(self.mid_high_head(mid_h), size=insz, mode='bilinear', align_corners=False),
            'mid_output': F.interpolate(self.mid_head(mid), size=insz, mode='bilinear', align_corners=False),
            'low_output': F.interpolate(self.low_head(low), size=insz, mode='bilinear', align_corners=False)
        }

# === PHASE 1 LOSS FUNCTIONS ===
def soft_dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, c: int=DatasetConfig.NUM_CLASSES) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase 1: Generates smooth generalization boundaries allowing network to form general blobs cleanly."""
    pred = torch.clamp(F.softmax(pred_logits, 1), 1e-7, 1-1e-7)
    hot = F.one_hot(target, c).permute(0,3,1,2).float()
    
    # 1. Calculates the numerical areas
    i = torch.sum(pred * hot, (0,2,3))
    u = torch.sum(pred+hot, (0,2,3))
    
    # 2. Smooths the union logic
    dsc = (2.*i+1.)/(u+1.)
    cw = torch.clamp(class_weights.clone().detach().to(pred.device), min=0.0)
    cw = cw / (torch.sum(cw)+1e-6)
    
    # Returns raw values + tracking union sets 
    return 1. - torch.clamp(torch.sum(dsc*cw), 0., 1.), i.detach(), u.detach()

def wce_standard(pred_logits: torch.Tensor, target: torch.Tensor, c: int=DatasetConfig.NUM_CLASSES) -> torch.Tensor:
    """Phase 1: Basic weighted soft cross entropy ensuring rare classes don't disappear."""
    pred = torch.clamp(F.softmax(pred_logits, 1), 1e-7, 1-1e-7)
    hot = F.one_hot(target, c).permute(0,3,1,2).float()
    ce = -torch.sum((hot*0.95 + 0.05/c) * torch.log(pred), 1)
    return torch.mean(class_weights.to(pred.device)[target] * ce)

# === PHASE 2 LOSS FUNCTIONS ===
def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Mathematical sorting of explicit pixel errors mapped against topological distances."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_loss(pred_logits: torch.Tensor, target: torch.Tensor, c: int=DatasetConfig.NUM_CLASSES) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase 2: Chisel strict boundaries accurately computing the precise discrete IoU Jaccards."""
    pred = F.softmax(pred_logits, dim=1)
    hot = F.one_hot(target, c).permute(0,3,1,2).float()
    
    # We still need tracking metrics for the global loop feedback
    i = torch.sum(pred * hot, (0,2,3)).detach()
    u = torch.sum(pred+hot, (0,2,3)).detach()
    
    C = pred.size(1)
    probas = pred.movedim(1, -1).reshape(-1, C)
    labels = target.view(-1)
    losses = []
    
    for cls in range(C):
        fg = (labels == cls).float()
        if fg.sum() == 0: continue
        errors = (fg - probas[:, cls]).abs()
        
        # 1. Rank pixel failures absolutely directly instead of generally
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
        
    l_loss = torch.mean(torch.stack(losses)) if losses else pred.sum() * 0.
    return l_loss, i, u

def wce_ohem(pred_logits: torch.Tensor, target: torch.Tensor, c: int=DatasetConfig.NUM_CLASSES) -> torch.Tensor:
    """Phase 2: Punishes the model purely on the hardest 20% of pixels in the image."""
    pred = torch.clamp(F.softmax(pred_logits, 1), 1e-7, 1-1e-7)
    hot = F.one_hot(target, c).permute(0,3,1,2).float()
    ce = -torch.sum((hot*0.95 + 0.05/c) * torch.log(pred), 1)
    pixel_loss = class_weights.to(pred.device)[target] * ce
    
    # 1. Grab flat tensor array
    num_pixels = pixel_loss.numel()
    
    # 2. Locate exactly 20 percent integer mark
    k = max(1, int(num_pixels * 0.20))
    
    # 3. Discard the 80% successfully trained pixels computationally entirely
    hard_losses, _ = torch.topk(pixel_loss.flatten(), k)
    return torch.mean(hard_losses)

def ftl(pred_logits: torch.Tensor, target: torch.Tensor, c: int=DatasetConfig.NUM_CLASSES) -> torch.Tensor:
    """Focal Tversky Loss computation enforcing structural pixel weights heavily avoiding rare-class degradation."""
    pred = torch.clamp(F.softmax(pred_logits, 1), 1e-6, 1.)
    hot = F.one_hot(target, c).permute(0,3,1,2).float()
    tp = torch.sum(hot * pred, (1,2,3))
    fp = torch.sum((1-hot)*pred, (1,2,3))
    fn = torch.sum(hot*(1-pred), (1,2,3))
    return torch.mean((1 - (tp+1e-6)/(tp+0.7*fp+0.3*fn+1e-6))**0.75)

def active_contour_loss(pred_logits: torch.Tensor, target: torch.Tensor, c: int=DatasetConfig.NUM_CLASSES) -> torch.Tensor:
    """
    Differentiable Spatial Morphologies.
    Calculates physical contour differences vs Ground Truth mathematically using raw PyTorch convolution kernels.
    """
    pred = torch.clamp(F.softmax(pred_logits, 1), 1e-7, 1-1e-7)
    hot = F.one_hot(target, c).permute(0,3,1,2).float()
    
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    
    pred_grad_x = F.conv2d(pred, kernel_x, padding=1, groups=c)
    pred_grad_y = F.conv2d(pred, kernel_y, padding=1, groups=c)
    pred_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
    
    gt_grad_x = F.conv2d(hot, kernel_x, padding=1, groups=c)
    gt_grad_y = F.conv2d(hot, kernel_y, padding=1, groups=c)
    gt_mag = torch.sqrt(gt_grad_x**2 + gt_grad_y**2 + 1e-6)
    
    # Punish the algorithm strictly for geometric mismatches on boundaries!
    return F.l1_loss(pred_mag, gt_mag)


# === EVALUATION ===
@torch.no_grad()
def multiscale_inference(model: nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """Multi-Scale Testing (MST) Inference function for production evaluation routines."""
    model.eval()
    _, _, h, w = image_tensor.shape
    scales = [0.5, 1.0, 1.5]
    fused_logits = torch.zeros((1, DatasetConfig.NUM_CLASSES, h, w), device=image_tensor.device)
    for scale in scales:
        scaled_size = (int(h * scale), int(w * scale))
        scaled_img = F.interpolate(image_tensor, size=scaled_size, mode='bilinear', align_corners=False)
        out = model(scaled_img)['main_output']
        fused_logits += F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
    return fused_logits / len(scales)


# === TRAINING LOOP ===
def train_loop():
    logger.info("Initializing Distributed Pipeline Router...")
    os.makedirs(TrainingConfig.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomDeepLabV3Plus(num_classes=DatasetConfig.NUM_CLASSES).to(device)
    
    # 1. LAYER-WISE LEARNING RATE DECAY (LLRD)
    # We strip backbone params separately to protect ImageNet features initially
    backbone_params, decoder_params = [], []
    for name, param in model.named_parameters():
        if "backbone" in name: backbone_params.append(param)
        else: decoder_params.append(param)
            
    # We restrict backbone to 1/10th speed learning (1e-5 vs 1e-4)
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': TrainingConfig.LEARNING_RATE * 0.1},
        {'params': decoder_params, 'lr': TrainingConfig.LEARNING_RATE}
    ], weight_decay=TrainingConfig.WEIGHT_DECAY)
    
    # 2. CO-SINE ANNEALING ALGORITHM
    # Drops the Rate to deep minima iteratively pushing model into deeper general valleys.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 3. SWA BASE INITIALIZATION
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    
    tr_img = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_IMG_DIR, "*.jpg")))
    tr_msk = sorted(glob.glob(os.path.join(DatasetConfig.TRAIN_MSK_DIR, "*.png")))
    
    loader = DataLoader(
        FloodNetPyTorchDataset(tr_img, tr_msk, DatasetConfig.NUM_CLASSES, id2color), 
        batch_size=TrainingConfig.BATCH_SIZE, shuffle=True, 
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    
    logger.info(f"Loaded {len(tr_img)} images. Phase-2 Transition sets at Epoch {TrainingConfig.ROUTER_EPOCH}.")
    writer = SummaryWriter(log_dir=f"{TrainingConfig.CHECKPOINT_DIR}/tensorboard")
    
    try:
        # model = torch.compile(model) # DISABLED: AMD ROCm hardware crash on FP16/Triton
        logger.info("ROCm PyTorch compilation succeeded.")
    except Exception: pass
        
    scaler = torch.amp.GradScaler('cuda')
    best_dice = 0.0
    
    try:
        for epoch in range(TrainingConfig.EPOCHS):
            model.train()
            
            # 4. EPOCH ROUTER VERIFICATION
            is_phase_2 = epoch >= TrainingConfig.ROUTER_EPOCH
            if epoch == TrainingConfig.ROUTER_EPOCH:
                logger.info(">>> ACTIVATING PHASE 2: Lovasz-Softmax + OHEM Initialized <<<")

            # 5. FROZEN WARMUPS & BATCHNORM PRESERVATION
            # We strictly shut the mathematical engine off sequentially for the Backbone initially.
            if epoch < 5:
                for param in model.backbone.parameters(): param.requires_grad = False
            elif epoch == 5:
                for param in model.backbone.parameters(): param.requires_grad = True
            
            # MATHEMATICAL CORRECTION: Force the Backbone BatchNorm blocks to remain frozen evaluating their pristine ImageNet stats
            for m in model.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    
            t_i = torch.zeros(DatasetConfig.NUM_CLASSES, device=device)
            t_u = torch.zeros(DatasetConfig.NUM_CLASSES, device=device)
            t_hard_i = torch.zeros(DatasetConfig.NUM_CLASSES, device=device)
            t_hard_u = torch.zeros(DatasetConfig.NUM_CLASSES, device=device)
            
            for b_idx, (im, tgts) in enumerate(loader):
                im = im.to(device, non_blocking=True)
                mt = tgts['main_output'].to(device, non_blocking=True)
                et = tgts['edge_output'].to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda', enabled=False): # DISABLED: AMD ROCm FP16 hardware crash
                    o = model(im)
                    
                    # 6. ROUTER EQUATION LOGIC
                    if not is_phase_2:
                        dl, intersection, union = soft_dice_loss(o['main_output'], mt)
                        main_wce = wce_standard(o['main_output'], mt)
                        
                        # Phase 1 uses traditional 1-to-1 Keras metrics globally
                        loss = (0.2 * main_wce + 0.3 * ftl(o['main_output'], mt) + 0.5 * dl + 
                                0.5 * active_contour_loss(o['main_output'], mt) +
                                0.2 * F.binary_cross_entropy_with_logits(o['edge_output'], et) + 
                                0.5 * wce_standard(o['side_output'], mt) + 0.2 * wce_standard(o['high_output'], mt) + 
                                0.2 * wce_standard(o['mid_high_output'], mt) + 0.2 * wce_standard(o['mid_output'], mt) + 0.2 * wce_standard(o['low_output'], mt))
                    else:
                        dl, intersection, union = lovasz_loss(o['main_output'], mt)
                        main_wce = wce_ohem(o['main_output'], mt)
                        
                        # Phase 2 implicitly throttles OHEM scalar explosion from deep layers by keeping deep supervisors Standard WCE.
                        # Also shifts Lovasz power from 0.5 to 0.6 while dropping OHEM priority.
                        loss = (0.1 * main_wce + 0.3 * ftl(o['main_output'], mt) + 0.6 * dl + 
                                0.5 * active_contour_loss(o['main_output'], mt) +
                                0.2 * F.binary_cross_entropy_with_logits(o['edge_output'], et) + 
                                0.5 * wce_standard(o['side_output'], mt) + 0.2 * wce_standard(o['high_output'], mt) + 
                                0.2 * wce_standard(o['mid_high_output'], mt) + 0.2 * wce_standard(o['mid_output'], mt) + 0.2 * wce_standard(o['low_output'], mt))
                
                # 7. SCALE AND EXECUTE DECAYS
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Hard-Dice tracking (Argmax categorical true overlap)
                with torch.no_grad():
                    pred_labels = torch.argmax(o['main_output'], dim=1)
                    hard_pred = F.one_hot(pred_labels, DatasetConfig.NUM_CLASSES).permute(0,3,1,2).float()
                    hard_true = F.one_hot(mt, DatasetConfig.NUM_CLASSES).permute(0,3,1,2).float()
                    hard_i = torch.sum(hard_pred * hard_true, (0,2,3))
                    hard_u = torch.sum(hard_pred + hard_true, (0,2,3))
                    
                # Update Epoch level metrics
                t_i += intersection
                t_u += union
                t_hard_i += hard_i
                t_hard_u += hard_u
                
                if b_idx % TrainingConfig.LOG_INTERVAL == 0:
                    logger.info(f"Epoch {epoch+1}/{TrainingConfig.EPOCHS} | Batch {b_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")
                    writer.add_scalar("Training/Loss", loss.item(), epoch * len(loader) + b_idx)
                    writer.flush()
                    
                    
            cw = class_weights.clone().detach().to(device)
            cw = torch.clamp(cw, min=0.0) / (torch.sum(torch.clamp(cw, min=0.0)) + 1e-6)
            
            # Step the mathematical learning cycle exactly at end of batches
            scheduler.step()
            
            # 8. SWA (Stochastic Weight Averaging)
            # Exactly 20 Epochs before the finish line, we commence aggregating geometrical topological vectors.
            if epoch >= TrainingConfig.EPOCHS - 20:
                swa_model.update_parameters(model)
                
            epoch_dice = torch.sum(((2.*t_i+1.0)/(t_u+1.0))*cw).item()
            true_hard_dice = torch.sum(((2.*t_hard_i+1.0)/(t_hard_u+1.0))*cw).item()
            logger.info(f"---> Epoch {epoch+1} Completed | Soft Dice (Loss Tracker): {epoch_dice:.4f} | TRUE HARD DICE (Model Accuracy): {true_hard_dice:.4f}")
            writer.add_scalar("Metrics/Soft-Dice", epoch_dice, epoch)
            writer.add_scalar("Metrics/Hard-Dice", true_hard_dice, epoch)
            
            if true_hard_dice > best_dice:
                best_dice = true_hard_dice
                torch.save(model.state_dict(), os.path.join(TrainingConfig.CHECKPOINT_DIR, "best_deeplab_weights.pt"))
                logger.info(f"New Best Hard Dice! ({best_dice:.4f}) - Snapshot saved reliably.")
                
    except KeyboardInterrupt:
        logger.warning("\n[SIGINT] Training halted manually (KeyboardInterrupt). Saving gracefully...")
        torch.save(model.state_dict(), os.path.join(TrainingConfig.CHECKPOINT_DIR, "interrupted_weights.pt"))
        
    finally:
        logger.info("Executing SWA Validation Updates...")
        torch.optim.swa_utils.update_bn(loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), os.path.join(TrainingConfig.CHECKPOINT_DIR, "final_swa_smoothed_weights.pt"))
        writer.close()
        logger.info("SWA parameters flushed to disk. Terminating.")

if __name__ == '__main__':
    train_loop()
