#!/bin/bash
# Master script to sequentially retrain all base models and the Synergistic model with SE Channel Attention

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export MIOPEN_LOG_LEVEL=3
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

LOG_FILE="master_se_retraining.log"

echo "========================================================" | tee -a $LOG_FILE
echo "Starting Squeeze-and-Excitation Retraining Pipeline" | tee -a $LOG_FILE
echo "Timestamp: $(date)" | tee -a $LOG_FILE
echo "========================================================" | tee -a $LOG_FILE

echo "[1/4] Starting DeepLabV3+ Retraining..." | tee -a $LOG_FILE
sg render -c ".venv/bin/python optimized_pytorch_version.py" >> $LOG_FILE 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: DeepLabV3+ retraining failed. Check $LOG_FILE for details." | tee -a $LOG_FILE
    exit 1
fi
echo "DeepLabV3+ retraining complete." | tee -a $LOG_FILE

echo "[2/4] Starting UNet Retraining..." | tee -a $LOG_FILE
sg render -c ".venv/bin/python train_unet.py" >> $LOG_FILE 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: UNet retraining failed. Check $LOG_FILE for details." | tee -a $LOG_FILE
    exit 1
fi
echo "UNet retraining complete." | tee -a $LOG_FILE

echo "[3/4] Starting FCN Retraining..." | tee -a $LOG_FILE
sg render -c ".venv/bin/python train_fcn.py" >> $LOG_FILE 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: FCN retraining failed. Check $LOG_FILE for details." | tee -a $LOG_FILE
    exit 1
fi
echo "FCN retraining complete." | tee -a $LOG_FILE

echo "[4/4] Starting Synergistic Ensemble Retraining..." | tee -a $LOG_FILE
# Force batch size 2 to prevent GPU OOM crashes on fusion training
export BATCH_SIZE=2
export ACCUM_ITER=4
sg render -c ".venv/bin/python train_synergistic.py" >> $LOG_FILE 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Synergistic training failed. Check $LOG_FILE for details." | tee -a $LOG_FILE
    exit 1
fi
echo "Synergistic Ensemble retraining complete." | tee -a $LOG_FILE

echo "========================================================" | tee -a $LOG_FILE
echo "All SE-enabled training pipelines completed successfully!" | tee -a $LOG_FILE
echo "Timestamp: $(date)" | tee -a $LOG_FILE
echo "========================================================" | tee -a $LOG_FILE
