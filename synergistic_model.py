import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_version import StandardUNet
from optimized_pytorch_version import CustomDeepLabV3Plus
from fcn_version import ResNet50FCN

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class FloodNetSynergisticNet(nn.Module):
    """
    Synergistic Feature-Level Fusion Segmentation Network for FloodNet.
    Fuses intermediate high-resolution feature maps from three pre-trained base models:
      - U-Net Decoder (32 channels, 1/1 scale)
      - DeepLab ASPP Decoder (64 channels, 1/4 scale)
      - FCN Auxiliary Decoder (512 channels, 1/8 scale)
    Features are projected and interpolated to 1/1 scale, concatenated, 
    and then fused via a deep multi-layer CNN head.
    """
    def __init__(self, num_classes=10, use_se=True):
        super().__init__()
        self.num_classes = num_classes
        self.use_se = use_se
        
        # 1. Instantiate the three high-performance base models
        self.unet = StandardUNet(num_classes=num_classes, use_se=use_se)
        self.deeplab = CustomDeepLabV3Plus(num_classes=num_classes, use_se=use_se)
        self.fcn = ResNet50FCN(num_classes=num_classes, use_se=use_se)
        
        # 2. Projection layer for FCN features (512 channels -> 64 channels)
        self.fcn_proj = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 3. Channel Attention module
        self.channel_attention = ChannelAttention(160, ratio=16)
        
        # 4. Fusion Head
        # Inputs: UNet (32 ch) + DeepLab upsampled (64 ch) + FCN projected/upsampled (64 ch) = 160 channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(160, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        insz = x.shape[-2:]
        
        # 1. Run forward pass through base models
        unet_outs = self.unet(x)
        deeplab_outs = self.deeplab(x)
        fcn_outs = self.fcn(x)
        
        # 2. Extract feature maps
        f_unet = unet_outs['features']      # Shape (B, 32, H, W)
        f_dl = deeplab_outs['features']      # Shape (B, 64, H/4, W/4)
        f_fcn = fcn_outs['features']        # Shape (B, 512, H/8, W/8)
        
        # 3. Project and/or upsample features to 1/1 scale
        f_dl_up = F.interpolate(f_dl, size=insz, mode='bilinear', align_corners=False) # Shape (B, 64, H, W)
        
        f_fcn_proj = self.fcn_proj(f_fcn)   # Shape (B, 64, H/8, W/8)
        f_fcn_up = F.interpolate(f_fcn_proj, size=insz, mode='bilinear', align_corners=False) # Shape (B, 64, H, W)
        
        # 4. Concatenate feature maps
        stacked_features = torch.cat([f_unet, f_dl_up, f_fcn_up], dim=1) # Shape (B, 160, H, W)
        
        # Apply Channel Attention to dynamically weight spatial features
        attention_weights = self.channel_attention(stacked_features)
        stacked_features = stacked_features * attention_weights
        
        # 5. Pass through fusion head
        fused_main = self.fusion_conv(stacked_features)
        
        return {
            'main_output': fused_main,
            
            # Keep individual main outputs for debugging/compatibility/ensembling
            'unet_output': unet_outs['main_output'],
            'deeplab_output': deeplab_outs['main_output'],
            'fcn_output': fcn_outs['main_output']
        }
