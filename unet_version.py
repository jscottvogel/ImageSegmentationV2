import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels_x1, in_channels_x2, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_channels_x1 + in_channels_x2, out_channels)

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
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        # Pre-trained ImageNet Encoder for high DICE
        self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1).features
        
        # Layer 1: 24 channels (1/2 scale)
        # Layer 2: 48 channels (1/4 scale)
        # Layer 3: 64 channels (1/8 scale)
        # Layer 5: 160 channels (1/16 scale)
        # Layer 7: 1280 channels (1/32 scale)

        self.up1 = Up(1280, 160, 512)
        self.up2 = Up(512, 64, 256)
        self.up3 = Up(256, 48, 128)
        self.up4 = Up(128, 24, 64)
        
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DoubleConv(64, 32)
        )
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        features = []
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i in [1, 2, 3, 5, 7]:
                features.append(x)
                
        # features[4] = 1/32, features[3] = 1/16, etc.
        x = self.up1(features[4], features[3])
        x = self.up2(x, features[2])
        x = self.up3(x, features[1])
        x = self.up4(x, features[0])
        x = self.up5(x)
        
        logits = self.outc(x)
        return {'main_output': logits}
