import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

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

class ResNet50FCN(nn.Module):
    """
    Heavyweight Fully Convolutional Network with ResNet50 Backbone.
    Replaces the lightweight MobileNetV3 to provide massive feature extraction 
    capabilities and auxiliary loss routing, bringing it on par with DeepLabV3+.
    Supports optional SE attention blocks at the end of each ResNet stage.
    """
    def __init__(self, num_classes=10, use_se=True):
        super().__init__()
        self.use_se = use_se
        # Load the pretrained FCN ResNet50
        self.model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
        
        # FCN-ResNet50 has a main classifier and an auxiliary classifier
        # The 4th index is the final Conv2d layer
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
        # Auxiliary heads for consistent multi-scale training
        self.side_head = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.high_head = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.mid_high_head = nn.Conv2d(512, num_classes, kernel_size=1)
        self.mid_head = nn.Conv2d(256, num_classes, kernel_size=1)
        self.low_head = nn.Conv2d(256, num_classes, kernel_size=1)

        # Squeeze-and-Excitation attention blocks for stages
        self.layer1_se = SEBlock(256) if use_se else nn.Identity()
        self.layer2_se = SEBlock(512) if use_se else nn.Identity()
        self.layer3_se = SEBlock(1024) if use_se else nn.Identity()
        self.layer4_se = SEBlock(2048) if use_se else nn.Identity()

    def forward(self, x):
        insz = x.shape[-2:]
        # Run backbone manually to get layer outputs
        # conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        backbone = self.model.backbone
        
        x_feat = backbone.conv1(x)
        x_feat = backbone.bn1(x_feat)
        x_feat = backbone.relu(x_feat)
        x_feat = backbone.maxpool(x_feat)
        
        layer1_out = self.layer1_se(backbone.layer1(x_feat))        # 256 channels, scale 1/4
        layer2_out = self.layer2_se(backbone.layer2(layer1_out))      # 512 channels, scale 1/8
        layer3_out = self.layer3_se(backbone.layer3(layer2_out))      # 1024 channels, scale 1/8 (dilated)
        layer4_out = self.layer4_se(backbone.layer4(layer3_out))      # 2048 channels, scale 1/8 (dilated)
        
        # Now pass to classifier and aux_classifier of the original fcn model
        feats = layer4_out
        for idx in range(4):
            feats = self.model.classifier[idx](feats)
        out = self.model.classifier[4](feats)
        aux_out = self.model.aux_classifier(layer3_out)
        
        return {
            'main_output': F.interpolate(out, size=insz, mode='bilinear', align_corners=False),
            'features': feats,
            'aux_output': F.interpolate(aux_out, size=insz, mode='bilinear', align_corners=False),
            'side_output': F.interpolate(self.side_head(layer4_out), size=insz, mode='bilinear', align_corners=False),
            'high_output': F.interpolate(self.high_head(layer3_out), size=insz, mode='bilinear', align_corners=False),
            'mid_high_output': F.interpolate(self.mid_high_head(layer2_out), size=insz, mode='bilinear', align_corners=False),
            'mid_output': F.interpolate(self.mid_head(layer1_out), size=insz, mode='bilinear', align_corners=False),
            'low_output': F.interpolate(self.low_head(layer1_out), size=insz, mode='bilinear', align_corners=False)
        }

