import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
os.environ["MIOPEN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

import torch

if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    from optimized_pytorch_version import TrainingConfig
    TrainingConfig.EPOCHS = 1
    TrainingConfig.ROUTER_EPOCH = 1
    TrainingConfig.LOG_INTERVAL = 1  # Log every batch to observe training speed immediately
    TrainingConfig.EPOCHS_OVERRIDDEN = True

    from train_synergistic import train_synergistic

    print("Starting quick 1-epoch sanity training check...")
    train_synergistic()
    print("Sanity training check completed successfully!")
