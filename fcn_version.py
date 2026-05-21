import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

class ResNet50FCN(nn.Module):
    """
    Heavyweight Fully Convolutional Network with ResNet50 Backbone.
    Replaces the lightweight MobileNetV3 to provide massive feature extraction 
    capabilities and auxiliary loss routing, bringing it on par with DeepLabV3+.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Load the pretrained FCN ResNet50
        self.model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
        
        # FCN-ResNet50 has a main classifier and an auxiliary classifier
        # The 4th index is the final Conv2d layer
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        out = self.model(x)
        # We route 'out' to 'main_output' and 'aux' to 'aux_output'
        return {
            'main_output': out['out'],
            'aux_output': out['aux']
        }
