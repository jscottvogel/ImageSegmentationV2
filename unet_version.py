import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm => ReLU) * 2 with optional SEBlock"""
    def __init__(self, in_channels, out_channels, use_se=False):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        return self.se(self.double_conv(x))

class Up(nn.Module):
    def __init__(self, in_channels_x1, in_channels_x2, out_channels, use_se=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_channels_x1 + in_channels_x2, out_channels, use_se=use_se)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class StandardUNet(nn.Module):
    """
    Enterprise UNet Implementation targeting Image Segmentation with EfficientNetV2 Backbone.
    Supports optional SE attention blocks in the decoder.
    """
    def __init__(self, num_classes=10, use_se=True):
        super().__init__()
        self.num_classes = num_classes
        self.use_se = use_se

        # Pre-trained ImageNet Encoder for high DICE
        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1).features
        
        # Layer 1: 24 channels (1/2 scale)
        # Layer 2: 48 channels (1/4 scale)
        # Layer 3: 64 channels (1/8 scale)
        # Layer 5: 160 channels (1/16 scale)
        # Layer 7: 1280 channels (1/32 scale)

        self.up1 = Up(1280, 160, 512, use_se=use_se)
        self.up2 = Up(512, 64, 256, use_se=use_se)
        self.up3 = Up(256, 48, 128, use_se=use_se)
        self.up4 = Up(128, 24, 64, use_se=use_se)
        
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DoubleConv(64, 32, use_se=use_se)
        )
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # Auxiliary heads for consistent multi-scale training
        self.side_head = nn.Conv2d(1280, num_classes, kernel_size=1)
        self.high_head = nn.Conv2d(512, num_classes, kernel_size=1)
        self.mid_high_head = nn.Conv2d(256, num_classes, kernel_size=1)
        self.mid_head = nn.Conv2d(128, num_classes, kernel_size=1)
        self.low_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        insz = x.shape[-2:]
        features = []
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i in [1, 2, 3, 5, 7]:
                features.append(x)
                
        # features[4] = 1/32, features[3] = 1/16, etc.
        side_feat = features[4]
        
        x_up1 = self.up1(features[4], features[3])      # 512 channels, 1/16 scale
        x_up2 = self.up2(x_up1, features[2])            # 256 channels, 1/8 scale
        x_up3 = self.up3(x_up2, features[1])            # 128 channels, 1/4 scale
        x_up4 = self.up4(x_up3, features[0])            # 64 channels, 1/2 scale
        x_up5 = self.up5(x_up4)                         # 32 channels, 1/1 scale
        
        logits = self.outc(x_up5)
        
        return {
            'main_output': logits,
            'features': x_up5,
            'side_output': F.interpolate(self.side_head(side_feat), size=insz, mode='bilinear', align_corners=False),
            'high_output': F.interpolate(self.high_head(x_up1), size=insz, mode='bilinear', align_corners=False),
            'mid_high_output': F.interpolate(self.mid_high_head(x_up2), size=insz, mode='bilinear', align_corners=False),
            'mid_output': F.interpolate(self.mid_head(x_up3), size=insz, mode='bilinear', align_corners=False),
            'low_output': F.interpolate(self.low_head(x_up4), size=insz, mode='bilinear', align_corners=False)
        }

