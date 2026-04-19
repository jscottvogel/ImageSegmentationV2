import os
import tensorflow as tf
import torch
import torch.nn.functional as F
import numpy as np

class_weights_np = np.array([
    0.0481, 0.4213, 0.0392, 0.4079, 0.0262,
    0.0305, 0.0100, 0.0614, 0.0544, 0.0010
], dtype=np.float32)

class_weights_tf = tf.Variable(class_weights_np, trainable=False)

# ---- TF Loss Definition ----
def dice_coefficient_weighted_tf(y_true, y_pred, smooth=1e-6):
    num_classes = tf.shape(y_pred)[-1]
    y_true_flat = tf.reshape(y_true, [-1, num_classes])
    y_pred_flat = tf.reshape(y_pred, [-1, num_classes])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=0)
    union = tf.reduce_sum(y_true_flat + y_pred_flat, axis=0)
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)
    temp_class_weights = tf.clip_by_value(class_weights_tf, 0.0, tf.reduce_max(class_weights_tf))
    temp_class_weights /= tf.reduce_sum(temp_class_weights) + 1e-6
    weighted_dice = dice_per_class * temp_class_weights
    total_dice = tf.reduce_sum(weighted_dice)
    return tf.clip_by_value(total_dice, 0.0, 1.0)

def dice_loss_tf(y_true, y_pred, smooth=1e-6):
    dice = dice_coefficient_weighted_tf(y_true, y_pred, smooth=smooth)
    return 1.0 - tf.clip_by_value(dice, 0.0, 1.0)

def weighted_cce_tf(y_true, y_pred):
    weights = tf.reduce_sum(class_weights_tf * y_true, axis=-1)
    ce = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.05)(y_true, y_pred)
    return tf.reduce_mean(weights * ce)

def focal_tversky_loss_tf(alpha=0.7, beta=0.3, gamma=0.75):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0)
        tp = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2, 3])
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2, 3])
        tversky = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
        return tf.reduce_mean((1 - tversky) ** gamma)
    return loss

def combined_loss_tf(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    ce_loss = weighted_cce_tf(y_true, y_pred)
    d_loss = dice_loss_tf(y_true, y_pred)
    focal_loss = focal_tversky_loss_tf()(y_true, y_pred)
    return 0.2 * ce_loss + 0.3 * focal_loss + 0.5 * d_loss

# ---- PyTorch Loss Definition ----
def dice_loss_pt_weighted(pred, target, class_weights, num_classes=10):
    pred = torch.clamp(pred, 1e-7, 1.0 - 1e-7)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    intersection = torch.sum(pred * target_one_hot, dim=(0, 2, 3))
    union = torch.sum(pred + target_one_hot, dim=(0, 2, 3))
    dice_per_class = (2. * intersection + 1e-6) / (union + 1e-6)
    
    cw = class_weights.clone().detach()
    cw = torch.clamp(cw, min=0.0, max=torch.max(cw))
    cw_norm = cw / (torch.sum(cw) + 1e-6)
    weighted_dice = dice_per_class * cw_norm
    total_dice = torch.sum(weighted_dice)
    return 1.0 - torch.clamp(total_dice, 0.0, 1.0)

def weighted_cce_pt(pred, target, class_weights, num_classes=10):
    pred = torch.clamp(pred, 1e-7, 1.0 - 1e-7)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    # Keras label_smoothing=0.05 for one-hot is: y_true * (1 - 0.05) + 0.05 / num_classes
    target_smoothed = target_one_hot * (1.0 - 0.05) + 0.05 / num_classes
    
    # Ce calculation: -sum(target * log(pred)) over classes
    log_pred = torch.log(pred)
    ce_per_pixel = -torch.sum(target_smoothed * log_pred, dim=1) # shape (B, H, W)
    
    # Keras applies class weight per pixel: weight = sum(class_weights * y_true_one_hot, axis=-1)
    # NOTE: tf.reduce_sum(class_weights * y_true) matches the class_weights indexed by y_true
    # target has shape (B, H, W). cw_per_pixel is shape (B, H, W)
    cw_per_pixel = class_weights[target]
    
    weighted_ce = cw_per_pixel * ce_per_pixel
    return torch.mean(weighted_ce)

def focal_tversky_loss_pt(pred, target, num_classes=10, alpha=0.7, beta=0.3, gamma=0.75):
    pred = torch.clamp(pred, 1e-6, 1.0)
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    
    tp = torch.sum(target_one_hot * pred, dim=(1, 2, 3))
    fp = torch.sum((1 - target_one_hot) * pred, dim=(1, 2, 3))
    fn = torch.sum(target_one_hot * (1 - pred), dim=(1, 2, 3))
    
    tversky = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
    return torch.mean((1 - tversky) ** gamma)

def combined_loss_pt(pred, target, class_weights, num_classes=10):
    pred = F.softmax(pred, dim=1)
    ce_loss = weighted_cce_pt(pred, target, class_weights, num_classes)
    d_loss = dice_loss_pt_weighted(pred, target, class_weights, num_classes)
    f_loss = focal_tversky_loss_pt(pred, target, num_classes)
    return 0.2 * ce_loss + 0.3 * f_loss + 0.5 * d_loss


np.random.seed(42)
B, H, W, C = 2, 64, 64, 10
logits_np = np.random.randn(B, H, W, C).astype(np.float32)
labels_np = np.random.randint(0, C, size=(B, H, W))

y_pred_tf = tf.nn.softmax(logits_np, axis=-1)
y_true_tf = tf.one_hot(labels_np, C)

# calculate TF Loss
c_tf = combined_loss_tf(y_true_tf, y_pred_tf)
print(f"TF Combined Loss = {c_tf.numpy():.6f}")

# calculate PT Loss
class_weights_pt = torch.tensor(class_weights_np, dtype=torch.float32)
logits_pt = torch.tensor(logits_np).permute(0, 3, 1, 2)
target_pt = torch.tensor(labels_np, dtype=torch.int64)

c_pt = combined_loss_pt(logits_pt, target_pt, class_weights_pt, C)
print(f"PyTorch Combined Loss = {c_pt.item():.6f}")
