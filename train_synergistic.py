import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import glob
import logging
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import random
from tqdm import tqdm

from synergistic_model import FloodNetSynergisticNet
from optimized_pytorch_version import (
    DatasetConfig, TrainingConfig, 
    id2color, soft_dice_loss, wce_standard, ftl, active_contour_loss, 
    lovasz_loss, wce_ohem, class_weights, rgb_to_mask, mask_to_soft_edge
)

def calc_loss(preds, labels, epoch):
    """Calculates hybrid loss based on current epoch phase."""
    is_phase_2 = epoch >= TrainingConfig.ROUTER_EPOCH
    
    if not is_phase_2:
        d_loss, _, _ = soft_dice_loss(preds, labels)
        wce_loss = wce_standard(preds, labels)
        f_loss = ftl(preds, labels)
        ac_loss = active_contour_loss(preds, labels)
        loss = (0.2 * wce_loss) + (0.3 * f_loss) + (0.4 * d_loss) + (0.1 * ac_loss)
    else:
        d_loss, _, _ = lovasz_loss(preds, labels)
        wce_loss = wce_ohem(preds, labels)
        f_loss = ftl(preds, labels)
        ac_loss = active_contour_loss(preds, labels)
        loss = (0.1 * wce_loss) + (0.3 * f_loss) + (0.6 * d_loss) + (0.2 * ac_loss)
        
    return loss

def calc_aux_loss(preds, labels):
    """Calculates lightweight loss for decoder heads to save memory and time."""
    d_loss, _, _ = soft_dice_loss(preds, labels)
    wce_loss = wce_standard(preds, labels)
    return (0.4 * wce_loss) + (0.6 * d_loss)

def apply_mosaic_cached(images, masks, edges, out_size):
    h, w = out_size
    yc, xc = [int(random.uniform(0.3, 0.7) * s) for s in out_size]
    
    output_img = np.zeros((h, w, 3), dtype=np.uint8)
    output_msk = np.zeros((h, w), dtype=np.int64)
    output_edg = np.zeros((h, w), dtype=np.float32)
    
    for i in range(4):
        img = images[i]
        msk = masks[i]
        edg = edges[i]
        if i == 0:
            output_img[:yc, :xc] = img[:yc, :xc]
            output_msk[:yc, :xc] = msk[:yc, :xc]
            output_edg[:yc, :xc] = edg[:yc, :xc]
        elif i == 1:
            output_img[:yc, xc:] = img[:yc, xc:]
            output_msk[:yc, xc:] = msk[:yc, xc:]
            output_edg[:yc, xc:] = edg[:yc, xc:]
        elif i == 2:
            output_img[yc:, :xc] = img[yc:, :xc]
            output_msk[yc:, :xc] = msk[yc:, :xc]
            output_edg[yc:, :xc] = edg[yc:, :xc]
        else:
            output_img[yc:, xc:] = img[yc:, xc:]
            output_msk[yc:, xc:] = msk[yc:, xc:]
            output_edg[yc:, xc:] = edg[yc:, xc:]
            
    return output_img, output_msk, output_edg

def apply_cutmix_cached(img1, msk1, edg1, img2, msk2, edg2):
    img1 = img1.copy()
    msk1 = msk1.copy()
    edg1 = edg1.copy()
    
    h, w = img1.shape[:2]
    cut_rat = np.sqrt(1. - np.random.beta(1.0, 1.0))
    cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
    cx, cy = random.randint(0, w), random.randint(0, h)

    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, w), np.clip(cy - cut_h // 2, 0, h)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, w), np.clip(cy + cut_h // 2, 0, h)

    img1[bby1:bby2, bbx1:bbx2] = img2[bby1:bby2, bbx1:bbx2]
    msk1[bby1:bby2, bbx1:bbx2] = msk2[bby1:bby2, bbx1:bbx2]
    edg1[bby1:bby2, bbx1:bbx2] = edg2[bby1:bby2, bbx1:bbx2]
    
    return img1, msk1, edg1

class FastFloodNetPyTorchDataset(Dataset):
    def __init__(self, image_paths, mask_paths, num_classes, id2color, apply_aug=True, use_mosaic=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.num_classes = num_classes
        self.id2color = id2color
        self.apply_aug = apply_aug
        self.use_mosaic = use_mosaic
        
        # Albumentations with additional_targets for synchronized edge map augmentation
        self.transform = A.Compose([
            A.Resize(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
            A.RandomBrightnessContrast(p=0.5), 
            A.Normalize(), 
            ToTensorV2()
        ], additional_targets={'edge': 'mask'}) if apply_aug else A.Compose([
            A.Resize(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH), 
            A.Normalize(), 
            ToTensorV2()
        ], additional_targets={'edge': 'mask'})
        
        # Precompute cache directory on disk to avoid RAM OOM crashes
        self.cache_dir = "preprocessed_cache"
        self.images_dir = os.path.join(self.cache_dir, "images")
        self.labels_dir = os.path.join(self.cache_dir, "labels")
        self.edges_dir = os.path.join(self.cache_dir, "edges")
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        os.makedirs(self.edges_dir, exist_ok=True)
        
        self.preprocessed_images = []
        self.preprocessed_labels = []
        self.preprocessed_edges = []
        
        print("Checking/generating preprocessed dataset cache on disk...")
        for idx in tqdm(range(len(image_paths)), desc="Disk Cache Check"):
            img_name = os.path.basename(image_paths[idx])
            base_name, _ = os.path.splitext(img_name)
            
            cache_img_path = os.path.join(self.images_dir, f"{base_name}.png")
            cache_label_path = os.path.join(self.labels_dir, f"{base_name}.png")
            cache_edge_path = os.path.join(self.edges_dir, f"{base_name}.png")
            
            if not (os.path.exists(cache_img_path) and os.path.exists(cache_label_path) and os.path.exists(cache_edge_path)):
                # Load and resize BGR image
                img = cv2.imread(image_paths[idx])
                img = cv2.resize(img, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(cache_img_path, img)
                
                # Load, resize, convert mask
                msk = cv2.cvtColor(cv2.imread(mask_paths[idx]), cv2.COLOR_BGR2RGB)
                msk = cv2.resize(msk, (DatasetConfig.IMG_WIDTH, DatasetConfig.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
                label = rgb_to_mask(msk, id2color, num_classes)
                cv2.imwrite(cache_label_path, label)
                
                # Generate edge
                edge = mask_to_soft_edge(label, num_classes)
                edge_uint8 = (edge * 255.0).astype(np.uint8)
                cv2.imwrite(cache_edge_path, edge_uint8)
                
            self.preprocessed_images.append(cache_img_path)
            self.preprocessed_labels.append(cache_label_path)
            self.preprocessed_edges.append(cache_edge_path)
            
        print("Caching preprocessed dataset in RAM...")
        cached_imgs = []
        cached_lbls = []
        cached_edgs = []
        for idx in tqdm(range(len(image_paths)), desc="RAM Cache Loading"):
            img = cv2.cvtColor(cv2.imread(self.preprocessed_images[idx]), cv2.COLOR_BGR2RGB)
            label = cv2.imread(self.preprocessed_labels[idx], cv2.IMREAD_GRAYSCALE)
            # Store edges as uint8 in RAM, convert to float32 on the fly in _load_preprocessed to save 75% memory
            edge = cv2.imread(self.preprocessed_edges[idx], cv2.IMREAD_GRAYSCALE)
            
            cached_imgs.append(torch.from_numpy(img))
            cached_lbls.append(torch.from_numpy(label))
            cached_edgs.append(torch.from_numpy(edge))
            
        # Stack into single large tensors to avoid pickling overhead (3 tensor objects instead of 5529 lists of tensors)
        self.cached_images = torch.stack(cached_imgs)
        self.cached_labels = torch.stack(cached_lbls)
        self.cached_edges = torch.stack(cached_edgs)
            
    def _load_preprocessed(self, idx):
        # Convert torch Tensors back to numpy arrays (zero-copy view) for Albumentations and CutMix/Mosaic
        img = self.cached_images[idx].numpy()
        label = self.cached_labels[idx].numpy()
        edge = self.cached_edges[idx].numpy().astype(np.float32) / 255.0
        return img, label, edge

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        if self.use_mosaic and self.apply_aug and random.random() < 0.3:
            idxs = [idx] + random.sample(range(len(self.image_paths)), 3)
            imgs_msks_edges = [self._load_preprocessed(i) for i in idxs]
            img, label, edge = apply_mosaic_cached(
                [im for im, _, _ in imgs_msks_edges], 
                [mk for _, mk, _ in imgs_msks_edges], 
                [ed for _, _, ed in imgs_msks_edges], 
                DatasetConfig.IMAGE_SIZE
            )
        elif self.use_mosaic and self.apply_aug and random.random() < 0.3:
            idx2 = random.randint(0, len(self.image_paths)-1)
            img1, msk1, edg1 = self._load_preprocessed(idx)
            img2, msk2, edg2 = self._load_preprocessed(idx2)
            img, label, edge = apply_cutmix_cached(img1, msk1, edg1, img2, msk2, edg2)
        else:
            img, label, edge = self._load_preprocessed(idx)
            
        augmented = self.transform(image=img, mask=label, edge=edge)
        img_tensor = augmented['image']
        label_tensor = augmented['mask'].long()
        edge_tensor = augmented['edge'].float().unsqueeze(0)
        
        return img_tensor, {'main_output': label_tensor, 'edge_output': edge_tensor}

def initialize_synergistic_weights(model, device, logger):
    checkpoint_dir = "model_checkpoint"
    
    # 0. Load entire synergistic model weights if they exist (for EXP-03 fine-tuning)
    synergistic_path = os.path.join(checkpoint_dir, "FloodNet_Synergistic", "best_synergistic_weights.pt")
    if os.path.exists(synergistic_path):
        logger.info(f"Initializing entire model from {synergistic_path}")
        state = torch.load(synergistic_path, map_location=device, weights_only=True)
        state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state.items() if k != "n_averaged"}
        # Load with strict=False to skip the new spatial_attention parameters
        model.load_state_dict(state, strict=False)
        logger.info("Loaded pre-trained synergistic weights successfully.")
        for h in logger.handlers: h.flush()
        return
        
    # 1. Load UNet weights
    unet_path = os.path.join(checkpoint_dir, "FloodNet_UNet", "best_unet_weights.pt")
    if os.path.exists(unet_path):
        logger.info(f"Initializing UNet from {unet_path}")
        unet_state = torch.load(unet_path, map_location=device, weights_only=True)
        unet_state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in unet_state.items() if k != "n_averaged"}
        model.unet.load_state_dict(unet_state, strict=False)
        logger.info("Loaded UNet weights successfully.")
    else:
        logger.warning(f"UNet weights not found at {unet_path}")
        
    # 2. Load DeepLab weights
    deeplab_path = os.path.join(checkpoint_dir, "FloodNet_PyTorch", "best_deeplab_weights.pt")
    if os.path.exists(deeplab_path):
        logger.info(f"Initializing DeepLab from {deeplab_path}")
        dl_state = torch.load(deeplab_path, map_location=device, weights_only=True)
        dl_state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in dl_state.items() if k != "n_averaged"}
        model.deeplab.load_state_dict(dl_state, strict=False)
        logger.info("Loaded DeepLab weights successfully.")
    else:
        logger.warning(f"DeepLab weights not found at {deeplab_path}")
        
    # 3. Load FCN weights
    fcn_path = os.path.join(checkpoint_dir, "FloodNet_FCN", "best_fcn_weights.pt")
    if os.path.exists(fcn_path):
        logger.info(f"Initializing FCN from {fcn_path}")
        fcn_state = torch.load(fcn_path, map_location=device, weights_only=True)
        fcn_state = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in fcn_state.items() if k != "n_averaged"}
        model.fcn.load_state_dict(fcn_state, strict=False)
        logger.info("Loaded FCN weights successfully.")
    else:
        logger.warning(f"FCN weights not found at {fcn_path}")
    for h in logger.handlers: h.flush()


def train_synergistic():
    log_file_name = "synergistic_training.log"
    checkpoint_dir = "model_checkpoint/FloodNet_Synergistic"
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    
    logger.propagate = False
    
    logger.info("FloodNetSynergisticNet Modular Fusion Training Pipeline Initialized")
    for h in logger.handlers: h.flush()
    
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    freeze_decoders = True  # Set to True for fast fusion-only fine-tuning in EXP-03
    if dry_run:
        TrainingConfig.EPOCHS = 1
        TrainingConfig.LOG_INTERVAL = 1
        checkpoint_dir = "model_checkpoint/FloodNet_Synergistic_DryRun"
    else:
        # Train for 3 epochs if decoders are frozen, otherwise 15 epochs
        TrainingConfig.EPOCHS = 3 if freeze_decoders else 15
    TrainingConfig.ROUTER_EPOCH = 0 # Use Phase 2 Lovasz/OHEM hybrid loss from the start

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    for h in logger.handlers: h.flush()
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 1. Instantiate the synergistic model
    model = FloodNetSynergisticNet(num_classes=DatasetConfig.NUM_CLASSES, use_se=True).to(device)
    
    # 2. Load weights from individual model checkpoints
    initialize_synergistic_weights(model, device, logger)
    
    # 3. Freeze only backbone parameters, unfreeze decoders/heads
    for param in model.parameters():
        param.requires_grad = True
        
    if freeze_decoders:
        logger.info("Freezing base model decoders (UNet, DeepLab, FCN) for fast fusion tuning.")
        for param in model.unet.parameters():
            param.requires_grad = False
        for param in model.deeplab.parameters():
            param.requires_grad = False
        for param in model.fcn.parameters():
            param.requires_grad = False
    else:
        for param in model.unet.backbone.parameters():
            param.requires_grad = False
        for param in model.deeplab.backbone.parameters():
            param.requires_grad = False
        for param in model.fcn.model.backbone.parameters():
            param.requires_grad = False
        
    # 4. Optimizer for fusion and decoders
    fusion_params = list(model.fusion_conv.parameters()) + list(model.fcn_proj.parameters()) + list(model.channel_attention.parameters()) + list(model.spatial_attention.parameters())
    decoder_params = [p for p in (list(model.unet.parameters()) + list(model.deeplab.parameters()) + list(model.fcn.parameters())) if p.requires_grad]
    
    trainable_params = fusion_params + decoder_params
    
    logger.info(f"Number of trainable parameters in decoders: {len(decoder_params)}")
    logger.info(f"Number of trainable parameters in fusion head: {len(fusion_params)}")
    
    param_groups = [{'params': fusion_params, 'lr': 5e-4}]
    if len(decoder_params) > 0:
        param_groups.append({'params': decoder_params, 'lr': 2e-5})
    
    optimizer = torch.optim.AdamW(
        param_groups, 
        weight_decay=TrainingConfig.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TrainingConfig.EPOCHS, eta_min=1e-6)
    
    # 5. Setup training dataset & loader
    T_IMG = DatasetConfig.TRAIN_IMG_DIR
    T_MSK = DatasetConfig.TRAIN_MSK_DIR
    
    tr_img = sorted(glob.glob(os.path.join(T_IMG, "*.jpg")))
    tr_msk = sorted(glob.glob(os.path.join(T_MSK, "*.png")))
    
    # 1. Separate the 150 validation images using the exact seed/selection as evaluate_synergistic.py
    np.random.seed(42)
    val_indices = set(np.random.choice(len(tr_img), 150, replace=False))
    
    # 2. Extract remaining training indices
    train_indices = [i for i in range(len(tr_img)) if i not in val_indices]
    
    # 3. Use all clean training split images
    tr_img = [tr_img[i] for i in train_indices]
    tr_msk = [tr_msk[i] for i in train_indices]
    
    logger.info(f"Found {len(tr_img)} training images in full clean training split.")
    for h in logger.handlers: h.flush()
    dataset = FastFloodNetPyTorchDataset(tr_img, tr_msk, DatasetConfig.NUM_CLASSES, id2color)
    
    batch_size = int(os.environ.get("BATCH_SIZE", "2"))
    accum_iter = int(os.environ.get("ACCUM_ITER", "4"))
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=0, 
        pin_memory=False
    )
    
    writer = SummaryWriter(log_dir=f"{checkpoint_dir}/tensorboard")
    best_dice = 0.0
    
    try:
        for epoch in range(TrainingConfig.EPOCHS):
            model.train()
            # Crucial: Keep frozen sub-models in eval mode so BatchNorm statistics are not corrupted!
            model.unet.eval()
            model.deeplab.eval()
            model.fcn.eval()
            
            t_hard_i = torch.zeros(DatasetConfig.NUM_CLASSES, device=device)
            t_hard_u = torch.zeros(DatasetConfig.NUM_CLASSES, device=device)
            epoch_loss = 0.0
            
            optimizer.zero_grad(set_to_none=True)
            limit_batches = int(os.environ.get("LIMIT_BATCHES", "0"))
            for b_idx, (images, targets) in enumerate(loader):
                if limit_batches > 0 and b_idx >= limit_batches:
                    logger.info(f"Reached LIMIT_BATCHES ({limit_batches}). Stopping epoch early.")
                    break
                try:
                    images = images.to(device, non_blocking=True)
                    labels = targets['main_output'].to(device, non_blocking=True)
                    
                    # Forward pass
                    preds_dict = model(images)
                    
                    # 1. Main fused output loss
                    loss = calc_loss(preds_dict['main_output'], labels, epoch)
                    
                    loss = loss / accum_iter
                    
                    loss.backward()
                    
                    if ((b_idx + 1) % accum_iter == 0) or (b_idx + 1 == len(loader)):
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=2.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        
                    epoch_loss += loss.item() * accum_iter
                    
                    # Calculate Dice metric on fused output
                    with torch.no_grad():
                        preds = preds_dict['main_output']
                        pred_labels = torch.argmax(preds, dim=1)
                        hard_pred = F.one_hot(pred_labels, DatasetConfig.NUM_CLASSES).permute(0,3,1,2).float()
                        
                        valid_mask_hard = (labels != 255).unsqueeze(1).float()
                        labels_safe = torch.where(labels == 255, torch.zeros_like(labels), labels)
                        hard_true = F.one_hot(labels_safe, DatasetConfig.NUM_CLASSES).permute(0,3,1,2).float() * valid_mask_hard
                        hard_pred = hard_pred * valid_mask_hard
                        
                        t_hard_i += torch.sum(hard_pred * hard_true, (0,2,3))
                        t_hard_u += torch.sum(hard_pred + hard_true, (0,2,3))
                        
                    if b_idx % TrainingConfig.LOG_INTERVAL == 0:
                        logger.info(f"Epoch {epoch + 1}/{TrainingConfig.EPOCHS} | Batch {b_idx}/{len(loader)} | Loss: {loss.item() * accum_iter:.4f}")
                        for h in logger.handlers: h.flush()
                        writer.add_scalar("Training/Loss", loss.item() * accum_iter, epoch * len(loader) + b_idx)
                        writer.flush()
                    
                    time.sleep(0.05)  # GPU cool-down delay to prevent thermal shutdowns / power spikes
                except Exception as e:
                    logger.error(f"CRITICAL CRASH ON EPOCH {epoch+1} BATCH {b_idx}")
                    logger.error(traceback.format_exc())
                    for h in logger.handlers: h.flush()
                    raise e
                    
            scheduler.step()
            
            # Weighted Dice accuracy check
            cw = torch.clamp(class_weights.clone().detach().to(device), min=0.0)
            cw = cw / (torch.sum(cw) + 1e-6)
            true_hard_dice = torch.sum(((2. * t_hard_i + 1.0) / (t_hard_u + 1.0)) * cw).item()
            
            metrics_avg = epoch_loss / len(loader)
            logger.info(f"---> Epoch {epoch + 1} Completed | True Hard Dice Accuracy: {true_hard_dice:.4f} | Avg Loss: {metrics_avg:.4f}")
            for h in logger.handlers: h.flush()
            writer.add_scalar("Metrics/Hard-Dice", true_hard_dice, epoch)
            writer.add_scalar("Metrics/Avg-Loss", metrics_avg, epoch)
            
            # Clear cache between epochs
            torch.cuda.empty_cache()
            
            if true_hard_dice > best_dice:
                best_dice = true_hard_dice
                torch.save(model.state_dict(), f"{checkpoint_dir}/best_synergistic_weights.pt")
                logger.info(f"New Best Synergistic Checkpoint Saved! (Dice: {best_dice:.4f})")
                for h in logger.handlers: h.flush()
                
    except KeyboardInterrupt:
        logger.warning(f"\n[SIGINT] Training halted manually (KeyboardInterrupt). Saving gracefully...")
        torch.save(model.state_dict(), f"{checkpoint_dir}/interrupted_weights.pt")
        for h in logger.handlers: h.flush()
        
    finally:
        writer.close()
        logger.info("Training complete.")
        for h in logger.handlers: h.flush()

if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    train_synergistic()
