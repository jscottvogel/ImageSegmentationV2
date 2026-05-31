#!/bin/bash
# Master training script for unpolluted model retraining on AMD GPU

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export MIOPEN_LOG_LEVEL=3

# Append to master log to preserve past history of DeepLabV3+ and UNet
echo "=== Resuming Master Unpolluted Retraining Pipeline (FCN & Meta-Ensemble) ===" >> master_training.log

# echo "[1/5] Starting DeepLabV3+ Training..." >> master_training.log
# sg render -c ".venv/bin/python optimized_pytorch_version.py" >> master_training.log 2>&1

# echo "[2/5] Starting UNet Training..." >> master_training.log
# sg render -c ".venv/bin/python train_unet.py" >> master_training.log 2>&1

echo "[3/5] Starting FCN Training..." >> master_training.log
sg render -c ".venv/bin/python train_fcn.py" >> master_training.log 2>&1

echo "[4/5] Starting Meta-Learner Ensemble Training..." >> master_training.log
sg render -c ".venv/bin/python train_meta_ensemble.py" >> master_training.log 2>&1

echo "[5/5] Starting Submission Generation..." >> master_training.log
sg render -c ".venv/bin/python generate_all_submissions.py" >> master_training.log 2>&1

echo "=== All training and inference steps completed successfully ===" >> master_training.log
