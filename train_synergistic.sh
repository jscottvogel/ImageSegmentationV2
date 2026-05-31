#!/bin/bash
# Script to train the synthesized FloodNetSynergisticNet model on AMD GPU

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export MIOPEN_LOG_LEVEL=3
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

echo "Starting FloodNetSynergisticNet Training..."
sg render -c ".venv/bin/python train_synergistic.py"
