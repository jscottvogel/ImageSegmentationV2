import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
import cv2
cv2.setNumThreads(0)
from standard_training_loop import run_standard_training
from fcn_version import ResNet50FCN

def train_fcn():
    run_standard_training(
        model_name="FCN",
        model_fn=ResNet50FCN,
        get_backbone_fn=lambda m: m.model.backbone,
        log_file_name="fcn_trace.log",
        checkpoint_dir="model_checkpoint/FloodNet_FCN"
    )

if __name__ == '__main__':
    train_fcn()
