import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from optimized_pytorch_version import CustomDeepLabV3Plus
from unet_version import StandardUNet
from fcn_version import ResNet50FCN

class FloodNetCompetitiveModel(nn.Module):
    """
    Unified Competitive Segmentation Model for FloodNet.
    Wraps UNet, FCN, and DeepLabV3+ under a single nn.Module and performs
    horizontal flip Test-Time Augmentation (TTA) internally during evaluation.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Instantiate the three base models
        self.unet = StandardUNet(num_classes=num_classes)
        self.fcn = ResNet50FCN(num_classes=num_classes)
        self.deeplab = CustomDeepLabV3Plus(num_classes=num_classes)
        
        # 2. Stacked Meta Layer (3 models * 10 classes = 30 input channels)
        self.meta_layer = nn.Conv2d(in_channels=30, out_channels=num_classes, kernel_size=1)
        
    def load_checkpoints(self, unet_path, fcn_path, deeplab_path, meta_path, device='cpu'):
        """Loads weights for all components, removing module/orig prefixes if present."""
        def clean_state_dict(path):
            state = torch.load(path, map_location=device, weights_only=True)
            if 'n_averaged' in state:
                del state['n_averaged']
            clean = {}
            for k, v in state.items():
                clean[k.replace('module.', '').replace('_orig_mod.', '')] = v
            return clean

        print(f"Loading UNet weights from: {unet_path}")
        self.unet.load_state_dict(clean_state_dict(unet_path), strict=False)
        
        print(f"Loading FCN weights from: {fcn_path}")
        self.fcn.load_state_dict(clean_state_dict(fcn_path), strict=False)
        
        print(f"Loading DeepLab weights from: {deeplab_path}")
        self.deeplab.load_state_dict(clean_state_dict(deeplab_path), strict=False)
        
        print(f"Loading Meta Layer weights from: {meta_path}")
        meta_state = torch.load(meta_path, map_location=device, weights_only=True)
        self.meta_layer.load_state_dict(meta_state)

    def forward(self, x):
        """
        Forward pass.
        If self.training is True: standard forward pass without TTA.
        If self.training is False: standard + horizontal flip TTA forward pass.
        """
        if self.training:
            # Standard forward pass
            out_unet = self.unet(x)
            if isinstance(out_unet, dict):
                out_unet = out_unet['main_output']
            p_unet = F.softmax(out_unet, dim=1)
            
            out_fcn = self.fcn(x)
            if isinstance(out_fcn, dict):
                out_fcn = out_fcn['main_output']
            p_fcn = F.softmax(out_fcn, dim=1)
            
            out_dl = self.deeplab(x)
            if isinstance(out_dl, dict):
                out_dl = out_dl['main_output']
            p_dl = F.softmax(out_dl, dim=1)
            
            # Concatenate probabilities along channel dimension
            stacked = torch.cat([p_unet, p_fcn, p_dl], dim=1)
            logits = self.meta_layer(stacked)
            return logits
        else:
            # Evaluation / Inference with built-in Horizontal Flip TTA
            # Run models sequentially to minimize peak VRAM usage
            
            # 1. UNet TTA pass
            out_unet_std = self.unet(x)
            if isinstance(out_unet_std, dict):
                out_unet_std = out_unet_std['main_output']
            p_unet_std = F.softmax(out_unet_std, dim=1)
            del out_unet_std
            
            x_flipped = torch.flip(x, dims=[3])
            out_unet_flip = self.unet(x_flipped)
            if isinstance(out_unet_flip, dict):
                out_unet_flip = out_unet_flip['main_output']
            p_unet_flip = F.softmax(out_unet_flip, dim=1)
            p_unet_unflip = torch.flip(p_unet_flip, dims=[3])
            del out_unet_flip, p_unet_flip
            
            p_unet_fused = (p_unet_std + p_unet_unflip) / 2.0
            del p_unet_std, p_unet_unflip
            
            # 2. FCN TTA pass
            out_fcn_std = self.fcn(x)
            if isinstance(out_fcn_std, dict):
                out_fcn_std = out_fcn_std['main_output']
            p_fcn_std = F.softmax(out_fcn_std, dim=1)
            del out_fcn_std
            
            out_fcn_flip = self.fcn(x_flipped)
            if isinstance(out_fcn_flip, dict):
                out_fcn_flip = out_fcn_flip['main_output']
            p_fcn_flip = F.softmax(out_fcn_flip, dim=1)
            p_fcn_unflip = torch.flip(p_fcn_flip, dims=[3])
            del out_fcn_flip, p_fcn_flip
            
            p_fcn_fused = (p_fcn_std + p_fcn_unflip) / 2.0
            del p_fcn_std, p_fcn_unflip
            
            # 3. DeepLab TTA pass
            out_dl_std = self.deeplab(x)
            if isinstance(out_dl_std, dict):
                out_dl_std = out_dl_std['main_output']
            p_dl_std = F.softmax(out_dl_std, dim=1)
            del out_dl_std
            
            out_dl_flip = self.deeplab(x_flipped)
            if isinstance(out_dl_flip, dict):
                out_dl_flip = out_dl_flip['main_output']
            p_dl_flip = F.softmax(out_dl_flip, dim=1)
            p_dl_unflip = torch.flip(p_dl_flip, dims=[3])
            del out_dl_flip, p_dl_flip, x_flipped
            
            p_dl_fused = (p_dl_std + p_dl_unflip) / 2.0
            del p_dl_std, p_dl_unflip
            
            # 4. Concatenate and pass through meta stacked layer
            stacked = torch.cat([p_unet_fused, p_fcn_fused, p_dl_fused], dim=1)
            del p_unet_fused, p_fcn_fused, p_dl_fused
            
            logits = self.meta_layer(stacked)
            del stacked
            return logits
