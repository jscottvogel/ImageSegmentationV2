import os
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DatasetConfig:
    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    NUM_CLASSES = 10
    IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

class TrainingConfig:
    BATCH_SIZE = 8
    EPOCHS = 101
    LEARNING_RATE = 0.0001
    CHECKPOINT_DIR = 'model_checkpoint/FloodNet_PyTorch'

id2color = {
    0: [0, 0, 0], 1: [255, 0, 0], 2: [200, 90, 90], 3: [128, 128, 0], 4: [155, 155, 155],
    5: [0, 255, 255], 6: [55, 0, 255], 7: [255, 0, 255], 8: [245, 245, 0], 9: [0, 255, 0],
}

def rgb_to_mask(rgb_arr, color_map, num_classes):
    from scipy.spatial import KDTree
    H, W, _ = rgb_arr.shape
    color_list = np.array(list(color_map.values()))
    class_indices = list(color_map.keys())
    tree = KDTree(color_list)
    reshaped = rgb_arr.reshape(-1, 3)
    _, indices = tree.query(reshaped)
    label_map = np.array([class_indices[i] for i in indices]).reshape(H, W)
    return label_map

def mask_to_soft_edge(label_map, num_classes):
    edge_maps = []
    for c in range(num_classes):
        class_mask = (label_map == c).astype(np.uint8)
        grad_x = cv2.Sobel(class_mask, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(class_mask, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_maps.append(grad_mag)
    edge = np.sum(edge_maps, axis=0)
    edge = np.clip(edge, 0, 1).astype(np.float32)
    return edge

def apply_mosaic(images, masks, out_size):
    h, w = out_size
    yc, xc = [int(random.uniform(0.3, 0.7) * s) for s in out_size]
    output_img = np.zeros((h, w, 3), dtype=np.uint8)
    output_msk = np.zeros((h, w), dtype=np.long)
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

class FloodNetPyTorchDataset(Dataset):
    def __init__(self, image_paths, mask_paths, num_classes, id2color, apply_aug=True, use_mosaic=True):
        self.image_paths, self.mask_paths = image_paths, mask_paths
        self.num_classes, self.id2color = num_classes, id2color
        self.apply_aug, self.use_mosaic = apply_aug, use_mosaic
        self.transform = A.Compose([
            A.Resize(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5), A.Normalize(), ToTensorV2()
        ]) if apply_aug else A.Compose([A.Resize(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH), A.Normalize(), ToTensorV2()])

    def __len__(self): return len(self.image_paths)

    def _load(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        msk = cv2.cvtColor(cv2.imread(self.mask_paths[idx]), cv2.COLOR_BGR2RGB)
        return img, rgb_to_mask(msk, self.id2color, self.num_classes)

    def __getitem__(self, idx):
        if self.use_mosaic and random.random() < 0.3:
            idxs = [idx] + random.sample(range(len(self.image_paths)), 3)
            imgs_msks = [self._load(i) for i in idxs]
            img, label = apply_mosaic([im for im, _ in imgs_msks], [mk for _, mk in imgs_msks], DatasetConfig.IMAGE_SIZE)
        else:
            img, label = self._load(idx)
        augmented = self.transform(image=img, mask=label)
        img, label = augmented['image'], augmented['mask'].numpy()
        edge = torch.tensor(mask_to_soft_edge(label, self.num_classes), dtype=torch.float32).unsqueeze(0)
        return img, {'main_output': torch.tensor(label, dtype=torch.long), 'edge_output': edge}

def conv_bn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.c3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.c4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.proj = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    def forward(self, x):
        size = x.shape[-2:]
        return self.proj(torch.cat([F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=False), self.c1(x), self.c2(x), self.c3(x), self.c4(x)], 1))

class CustomDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1).features
        self.aspp = ASPP(1280, 256)
        
        self.high_conv = conv_bn_relu(160, 128)
        self.fuse_high = conv_bn_relu(256 + 128, 256)
        self.mid_high_conv = conv_bn_relu(128, 128)
        self.fuse_mid_high = conv_bn_relu(256 + 128, 256)
        self.mid_conv = conv_bn_relu(64, 64)
        self.fuse_mid = conv_bn_relu(256 + 64, 128)
        self.low_conv = conv_bn_relu(48, 48)
        self.fuse_low = conv_bn_relu(128 + 48, 64)
        self.final_refine = conv_bn_relu(64, 64)
        
        self.main_head = nn.Conv2d(64, num_classes, 1)
        self.edge_proj = conv_bn_relu(48, 64)
        self.edge_head = nn.Conv2d(64, 1, 1)
        self.side_head = nn.Conv2d(256, num_classes, 1)
        self.high_head = nn.Conv2d(256, num_classes, 1)
        self.mid_high_head = nn.Conv2d(256, num_classes, 1)
        self.mid_head = nn.Conv2d(128, num_classes, 1)
        self.low_head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        insz = x.shape[-2:]
        fs = {}
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i==2: fs['low'] = x
            if i==3: fs['mid'] = x
            if i==4: fs['mid_h'] = x
            if i==5: fs['high'] = x
            if i==7: fs['top'] = x
                
        aspp = self.aspp(fs['top'])
        h = F.interpolate(aspp, size=fs['high'].shape[-2:], mode='bilinear', align_corners=False)
        high = self.fuse_high(torch.cat([h, self.high_conv(fs['high'])], 1))
        h = F.interpolate(high, size=fs['mid_h'].shape[-2:], mode='bilinear', align_corners=False)
        mid_h = self.fuse_mid_high(torch.cat([h, self.mid_high_conv(fs['mid_h'])], 1))
        h = F.interpolate(mid_h, size=fs['mid'].shape[-2:], mode='bilinear', align_corners=False)
        mid = self.fuse_mid(torch.cat([h, self.mid_conv(fs['mid'])], 1))
        h = F.interpolate(mid, size=fs['low'].shape[-2:], mode='bilinear', align_corners=False)
        low = self.fuse_low(torch.cat([h, self.low_conv(fs['low'])], 1))
        
        return {
            'main_output': F.interpolate(self.main_head(self.final_refine(low)), size=insz, mode='bilinear', align_corners=False),
            'edge_output': F.interpolate(self.edge_head(self.edge_proj(fs['low'])), size=insz, mode='bilinear', align_corners=False),
            'side_output': F.interpolate(self.side_head(aspp), size=insz, mode='bilinear', align_corners=False),
            'high_output': F.interpolate(self.high_head(high), size=insz, mode='bilinear', align_corners=False),
            'mid_high_output': F.interpolate(self.mid_high_head(mid_h), size=insz, mode='bilinear', align_corners=False),
            'mid_output': F.interpolate(self.mid_head(mid), size=insz, mode='bilinear', align_corners=False),
            'low_output': F.interpolate(self.low_head(low), size=insz, mode='bilinear', align_corners=False)
        }

class_weights = torch.tensor([0.0481, 0.4213, 0.0392, 0.4079, 0.0262, 0.0305, 0.0100, 0.0614, 0.0544, 0.0010], dtype=torch.float32)

def d_loss(pred, target, c=10):
    pred = torch.clamp(F.softmax(pred, 1), 1e-7, 1-1e-7)
    hot = F.one_hot(target, c).permute(0,3,1,2).float()
    i = torch.sum(pred * hot, (0,2,3))
    u = torch.sum(pred+hot, (0,2,3))
    dsc = (2.*i+1.)/(u+1.)
    cw = torch.clamp(class_weights.clone().detach().to(pred.device), min=0.0)
    cw = cw / (torch.sum(cw)+1e-6)
    return 1. - torch.clamp(torch.sum(dsc*cw), 0., 1.), i.detach(), u.detach()

def wce(pred, target, c=10):
    pred = torch.clamp(F.softmax(pred, 1), 1e-7, 1-1e-7)
    hot = F.one_hot(target, c).permute(0,3,1,2).float()
    ce = -torch.sum((hot*0.95 + 0.05/c) * torch.log(pred), 1)
    return torch.mean(class_weights.to(pred.device)[target] * ce)

def ftl(pred, target, c=10):
    pred = torch.clamp(F.softmax(pred, 1), 1e-6, 1.)
    hot = F.one_hot(target, c).permute(0,3,1,2).float()
    tp = torch.sum(hot * pred, (1,2,3))
    fp = torch.sum((1-hot)*pred, (1,2,3))
    fn = torch.sum(hot*(1-pred), (1,2,3))
    return torch.mean((1 - (tp+1e-6)/(tp+0.7*fp+0.3*fn+1e-6))**0.75)

def train_loop():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomDeepLabV3Plus().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LEARNING_RATE, weight_decay=1e-5)
    
    T_IMG = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/train/images"
    T_MSK = "/home/fred/Downloads/opencv-tf-project-3-image-segmentation-round-2/Project_3_FloodNet_Dataset/train/masks"
    tr_img = sorted(glob.glob(os.path.join(T_IMG, "*.jpg")))
    tr_msk = sorted(glob.glob(os.path.join(T_MSK, "*.png")))
    
    loader = DataLoader(FloodNetPyTorchDataset(tr_img, tr_msk, 10, id2color), batch_size=TrainingConfig.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    print(f"Training for {TrainingConfig.EPOCHS} Epochs on {len(tr_img)} images...")
    
    try:
        # model = torch.compile(model) # DISABLED: AMD ROCm hardware crash on FP16/Triton
    except Exception:
        pass
        
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(TrainingConfig.EPOCHS):
        model.train()
        t_i = torch.zeros(10, device=device)
        t_u = torch.zeros(10, device=device)
        
        for b_idx, (im, tgts) in enumerate(loader):
            im, mt, et = im.to(device, non_blocking=True), tgts['main_output'].to(device, non_blocking=True), tgts['edge_output'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=False): # DISABLED: AMD ROCm FP16 hardware crash
                o = model(im)
                dl, intersection, union = d_loss(o['main_output'], mt)
                loss = (0.2 * wce(o['main_output'], mt) + 0.3 * ftl(o['main_output'], mt) + 0.5 * dl + 
                        0.2 * F.binary_cross_entropy_with_logits(o['edge_output'], et) + 
                        0.5 * wce(o['side_output'], mt) + 0.2 * wce(o['high_output'], mt) + 
                        0.2 * wce(o['mid_high_output'], mt) + 0.2 * wce(o['mid_output'], mt) + 0.2 * wce(o['low_output'], mt))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            t_i += intersection
            t_u += union
            
            if b_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{TrainingConfig.EPOCHS} | Batch {b_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")
                
        cw = class_weights.clone().detach().to(device)
        cw = torch.clamp(cw, min=0.0) / (torch.sum(torch.clamp(cw, min=0.0)) + 1e-6)
        print(f"\n---> Epoch {epoch+1} Completed | Main Dice Score: {torch.sum(((2.*t_i+1.0)/(t_u+1.0))*cw).item():.4f}\n")

if __name__ == '__main__':
    train_loop()
