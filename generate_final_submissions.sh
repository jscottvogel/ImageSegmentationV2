#!/bin/bash
# Master script to generate all final submission candidates sequentially

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export MIOPEN_LOG_LEVEL=3
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

echo "========================================================"
echo "Starting Final Submission Generation Pipeline"
echo "Timestamp: $(date)"
echo "========================================================"

echo ""
echo "[1/3] Generating Blended Submissions (Full TTA, thresholds: t30, t95, t99)..."
.venv/bin/python generate_blended_submission.py

echo ""
echo "[2/3] Generating Final Ensemble Submissions (Mixed TTA, area thresholds: area128, area64)..."
.venv/bin/python generate_final_ensemble_submissions.py

echo ""
echo "[3/3] Generating Individual and Hybrid Submissions (No TTA)..."
.venv/bin/python generate_all_submissions.py

echo ""
echo "========================================================"
echo "All final submission candidate files generated successfully!"
echo "Timestamp: $(date)"
echo "========================================================"
