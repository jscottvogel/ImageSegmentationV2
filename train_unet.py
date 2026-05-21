import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import cv2
cv2.setNumThreads(0)
from standard_training_loop import run_standard_training
from unet_version import StandardUNet

def train_unet():
    run_standard_training(
        model_name="UNet",
        model_fn=StandardUNet,
        get_backbone_fn=lambda m: m.backbone,
        log_file_name="unet_trace.log",
        checkpoint_dir="model_checkpoint/FloodNet_UNet"
    )

if __name__ == '__main__':
    train_unet()
