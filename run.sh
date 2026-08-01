#!/bin/bash
set -o pipefail
cd /workspace/thesis_code
if python -u "nureal network/train_cv_robust.py" 2>&1 | tee training.log; then
    echo "Training completed successfully. Shutting down pod..."
    python shutdown_pod.py
else
    echo "Training failed. Pod will remain running for debugging."
fi
