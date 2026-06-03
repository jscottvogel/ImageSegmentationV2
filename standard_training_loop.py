import os
os.environ["MIOPEN_LOG_LEVEL"] = "3"  # Silence noisy MIOpen warnings to prevent log flooding
import torch
import glob
import logging
import traceback
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from optimized_pytorch_version import (
    FloodNetPyTorchDataset, DatasetConfig, TrainingConfig, 
    id2color, soft_dice_loss, wce_standard, ftl, active_contour_loss, 
    lovasz_loss, wce_ohem, class_weights
)

def run_standard_training(model_name, model_fn, get_backbone_fn, log_file_name, checkpoint_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_name),
            logging.StreamHandler()
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info(f"{model_name} Diagnostics Tracer Initialized")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Initializing Standard PyTorch {model_name} on {device}")
    
    dry_run = os.environ.get("LIMIT_BATCHES", "0") != "0" or os.environ.get("DRY_RUN", "0") == "1"
    if dry_run:
        TrainingConfig.EPOCHS = 1
        TrainingConfig.LOG_INTERVAL = 1
        
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = model_fn(num_classes=DatasetConfig.NUM_CLASSES).to(device)
    
    # Auto-Resume Logic
    best_ckpt = os.path.join(checkpoint_dir, f"best_{model_name.lower()}_weights.pt")
    if os.path.exists(best_ckpt):
        logger.info(f"Found existing checkpoint. Resuming {model_name} from: {best_ckpt}")
        try:
            state_dict = torch.load(best_ckpt, map_location=device, weights_only=True)
            model.load_state_dict({k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in state_dict.items()}, strict=False)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    
    backbone_params, decoder_params = [], []
    for name, param in model.named_parameters():
        if "backbone" in name: backbone_params.append(param)
        else: decoder_params.append(param)
        
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': TrainingConfig.LEARNING_RATE * 0.1},
        {'params': decoder_params, 'lr': TrainingConfig.LEARNING_RATE}
    ], weight_decay=TrainingConfig.WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TrainingConfig.EPOCHS, eta_min=1e-6)
    
    # FP16 (autocast) and GradScaler explicitly removed to prevent 
    # persistent 'HIP error: invalid device function' crashes on AMD GPUs.
    
    T_IMG = DatasetConfig.TRAIN_IMG_DIR
    T_MSK = DatasetConfig.TRAIN_MSK_DIR
    
    tr_img = sorted(glob.glob(os.path.join(T_IMG, "*.jpg")))
    tr_msk = sorted(glob.glob(os.path.join(T_MSK, "*.png")))
    
    ps_img = sorted(glob.glob(os.path.join(TrainingConfig.PSEUDO_IMG_DIR, "*.jpg")))
    ps_msk = sorted(glob.glob(os.path.join(TrainingConfig.PSEUDO_MSK_DIR, "*.png")))
    
    if TrainingConfig.USE_PSEUDO_LABELS and len(ps_msk) > 0:
        logger.info(f"Injecting {len(ps_msk)} high-confidence pseudo-masks into {model_name} training!")
        tr_img.extend(ps_img)
        tr_msk.extend(ps_msk)
    
    dataset = FloodNetPyTorchDataset(tr_img, tr_msk, DatasetConfig.NUM_CLASSES, id2color)
    loader = DataLoader(dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2, pin_memory=True, persistent_workers=True)
    
    writer = SummaryWriter(log_dir=f"{checkpoint_dir}/tensorboard")
    best_dice = 0.0
    accum_iter = TrainingConfig.GRAD_ACCUMULATION_STEPS
    
    try:
        for epoch in range(TrainingConfig.EPOCHS):
            model.train()
    
            backbone = get_backbone_fn(model)
            if epoch < 5:
                for param in backbone.parameters(): param.requires_grad = False
            elif epoch == 5:
                for param in backbone.parameters(): param.requires_grad = True
    
            for m in backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
    
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
    
                    # Standard FP32 inference without autocast
                    preds_dict = model(images)
                    preds = preds_dict['main_output']
    
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
                        
                    # Target Aux classifier (Crucial for deep ResNet-50 models to learn early representations)
                    if 'aux_output' in preds_dict:
                        aux_wce = wce_standard(preds_dict['aux_output'], labels)
                        loss = loss + 0.4 * aux_wce
    
                    loss = loss / accum_iter
                    
                    # Direct backward pass, no scaler
                    loss.backward()
    
                    if ((b_idx + 1) % accum_iter == 0) or (b_idx + 1 == len(loader)):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
    
                    epoch_loss += loss.item() * accum_iter
    
                    with torch.no_grad():
                        pred_labels = torch.argmax(preds, dim=1)
                        hard_pred = F.one_hot(pred_labels, DatasetConfig.NUM_CLASSES).permute(0,3,1,2).float()
                        
                        valid_mask_hard = (labels != 255).unsqueeze(1).float()
                        labels_safe = torch.where(labels == 255, torch.zeros_like(labels), labels)
                        hard_true = F.one_hot(labels_safe, DatasetConfig.NUM_CLASSES).permute(0,3,1,2).float() * valid_mask_hard
                        hard_pred = hard_pred * valid_mask_hard
                        
                        t_hard_i += torch.sum(hard_pred * hard_true, (0,2,3))
                        t_hard_u += torch.sum(hard_pred + hard_true, (0,2,3))
    
                    if b_idx % TrainingConfig.LOG_INTERVAL == 0:
                        logger.info(f"{model_name} Epoch {epoch + 1}/{TrainingConfig.EPOCHS} | Batch {b_idx}/{len(loader)} | Phase-{'2' if is_phase_2 else '1'} Loss: {loss.item() * accum_iter:.4f}")
                        writer.add_scalar("Training/Loss", loss.item() * accum_iter, epoch * len(loader) + b_idx)
                        writer.flush()
                except Exception as e:
                    logger.error(f"CRITICAL CRASH ON EPOCH {epoch+1} BATCH {b_idx}")
                    logger.error(traceback.format_exc())
                    logger.error(f"CRITICAL ERROR CAUGHT: See {log_file_name} for detailed traceback.")
                    raise e
    
            scheduler.step()
    
            cw = torch.clamp(class_weights.clone().detach().to(device), min=0.0)
            cw = cw / (torch.sum(cw) + 1e-6)
            true_hard_dice = torch.sum(((2. * t_hard_i + 1.0) / (t_hard_u + 1.0)) * cw).item()
    
            if epoch >= TrainingConfig.EPOCHS - 5:
                swa_model.update_parameters(model)
    
            metrics_avg = epoch_loss / len(loader)
            logger.info(f"---> {model_name} Epoch {epoch + 1} Completed | True Hard Dice Accuracy: {true_hard_dice:.4f} | Avg Loss: {metrics_avg:.4f}")
            writer.add_scalar("Metrics/Hard-Dice", true_hard_dice, epoch)
            writer.add_scalar("Metrics/Avg-Loss", metrics_avg, epoch)
    
            if true_hard_dice > best_dice:
                best_dice = true_hard_dice
                torch.save(model.state_dict(), f"{checkpoint_dir}/best_{model_name.lower()}_weights.pt")
                logger.info(f"New Best {model_name} Checkpoint Saved! (Dice: {best_dice:.4f})")
    
    except KeyboardInterrupt:
        logger.warning(f"\n[SIGINT] Training halted manually (KeyboardInterrupt). Saving gracefully...")
        torch.save(model.state_dict(), f"{checkpoint_dir}/interrupted_weights.pt")
        
    finally:
        dry_run = os.environ.get("LIMIT_BATCHES", "0") != "0" or os.environ.get("DRY_RUN", "0") == "1"
        if dry_run:
            logger.info("Skipping SWA BatchNorm updates during dry-run.")
            writer.close()
        else:
            logger.info("Executing final SWA BatchNorm Update for maximal test-set generalization on Pristine Data...")
            
            clean_swa_loader = DataLoader(
                FloodNetPyTorchDataset(tr_img, tr_msk, DatasetConfig.NUM_CLASSES, id2color, apply_aug=False), 
                batch_size=TrainingConfig.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2, pin_memory=True
            )
            
            torch.optim.swa_utils.update_bn(clean_swa_loader, swa_model, device=device)
            torch.save(swa_model.state_dict(), f"{checkpoint_dir}/final_swa_smoothed_{model_name.lower()}.pt")
            writer.close()
            logger.info(f"SWA {model_name} Weights successfully flushed to disk!")
