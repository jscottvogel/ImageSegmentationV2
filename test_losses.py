import torch
import torch.nn.functional as F
from optimized_pytorch_version import DatasetConfig, class_weights

def active_contour_loss(pred_logits: torch.Tensor, target: torch.Tensor, c: int=DatasetConfig.NUM_CLASSES) -> torch.Tensor:
    pred = torch.clamp(F.softmax(pred_logits, 1), 1e-7, 1-1e-7)
    
    valid_mask = (target != 255).unsqueeze(1).float()
    target_safe = torch.where(target == 255, torch.zeros_like(target), target)
    hot = F.one_hot(target_safe, c).permute(0,3,1,2).float()
    
    # Erode valid mask to ignore artificial boundaries
    eroded_valid_mask = 1 - F.max_pool2d(1 - valid_mask, kernel_size=3, stride=1, padding=1)
    
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    
    pred_grad_x = F.conv2d(pred, kernel_x, padding=1, groups=c)
    pred_grad_y = F.conv2d(pred, kernel_y, padding=1, groups=c)
    pred_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
    
    gt_grad_x = F.conv2d(hot, kernel_x, padding=1, groups=c)
    gt_grad_y = F.conv2d(hot, kernel_y, padding=1, groups=c)
    gt_mag = torch.sqrt(gt_grad_x**2 + gt_grad_y**2 + 1e-6)
    
    # Use eroded mask
    return F.l1_loss(pred_mag * eroded_valid_mask, gt_mag * eroded_valid_mask)

pred = torch.randn(2, 10, 16, 16)
target = torch.randint(0, 10, (2, 16, 16))
target[:, 8:12, 8:12] = 255

loss = active_contour_loss(pred, target)
print("Active contour loss:", loss.item())
