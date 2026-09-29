#!/bin/bash

#SBATCH --job-name=eigenscore
#SBATCH --gres=gpu:1
#SBATCH --partition=RTXA6Kq
#SBATCH --nodelist=node16
#SBATCH --output=/export/home2/sati0004/JBHI/BRIDGE/logs/%x_%j.out
#SBATCH --error=/export/home2/sati0004/JBHI/BRIDGE/logs/%x_%j.err

set -euo pipefail


# ============================================================
# ENVIRONMENT
# ============================================================

module purge

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate mech_interp


# ============================================================
# PATHS
# ============================================================

REPO_ROOT="$HOME/JBHI/BRIDGE"

SCRIPT="$REPO_ROOT/scripts/eigenscore_baseline.py"

GENERATION_ROOT="$REPO_ROOT/cache/baseline_generations"

BRIDGE_ROOT="$REPO_ROOT/outputs/probes"

OUTPUT_ROOT="$REPO_ROOT/outputs/baselines/eigenscore"

LOG_DIR="$REPO_ROOT/logs"


mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_ROOT"

cd "$REPO_ROOT"


# ============================================================
# MODEL ARGUMENT
# ============================================================

if [ "$#" -ne 1 ]; then

    echo "ERROR:"
    echo "Exactly one model is required."
    echo
    echo "Usage:"
    echo "sbatch slurm/eigenscore.sh MODEL"
    echo
    echo "Allowed:"
    echo "  biomistral"
    echo "  openbiollm"
    echo "  ultramedical"
    echo "  medgemma"
    echo "  lingshu"

    exit 1

fi


MODEL="$1"


case "$MODEL" in

    biomistral|openbiollm|ultramedical|medgemma|lingshu)
        ;;

    *)

        echo "ERROR:"
        echo "Unknown model: $MODEL"
        exit 1
        ;;

esac


# ============================================================
# RUN INFO
# ============================================================

echo "============================================================"
echo "BRIDGE INSIDE / EIGENSCORE"
echo "============================================================"
echo "Start time:       $(date)"
echo "Host:             $(hostname)"
echo "SLURM job ID:     ${SLURM_JOB_ID:-N/A}"
echo "Model:            $MODEL"
echo
echo "Repository root:  $REPO_ROOT"
echo "Script:           $SCRIPT"
echo "Generation root:  $GENERATION_ROOT"
echo "BRIDGE root:      $BRIDGE_ROOT"
echo "Output root:      $OUTPUT_ROOT"
echo "============================================================"
echo


# ============================================================
# GPU
# ============================================================

echo "============================================================"
echo "GPU INFORMATION"
echo "============================================================"

nvidia-smi

echo


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

echo "============================================================"
echo "PYTHON ENVIRONMENT"
echo "============================================================"

which python
python --version

python - <<'PY'

import sys
import torch
import transformers
import numpy
import pandas
import sklearn
import scipy

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("SciPy:", scipy.__version__)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA unavailable."
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

from transformers import (
    Qwen2_5_VLForConditionalGeneration
)

print(
    "Qwen2.5-VL import: PASS"
)

print(
    "Environment check: PASS"
)

PY

echo


# ============================================================
# SCRIPT CHECK
# ============================================================

if [ ! -f "$SCRIPT" ]; then

    echo "ERROR:"
    echo "Missing script:"
    echo "$SCRIPT"

    exit 1

fi


python -m py_compile "$SCRIPT"

echo "Syntax check: PASS"
echo


# ============================================================
# CACHE CHECK
# ============================================================

echo "============================================================"
echo "CHECKING GENERATION CACHE"
echo "============================================================"


HB_ROOT="$GENERATION_ROOT/$MODEL/healthbench/test"

VAL_ROOT="$GENERATION_ROOT/$MODEL/healthbench/validation"

CD_ROOT="$GENERATION_ROOT/$MODEL/clindet/external_test"


HB_COUNT=0
VAL_COUNT=0
CD_COUNT=0


if [ -d "$HB_ROOT" ]; then

    HB_COUNT=$(find "$HB_ROOT" \
        -type f \
        -name "*.pt" \
        | wc -l)

fi


if [ -d "$VAL_ROOT" ]; then

    VAL_COUNT=$(find "$VAL_ROOT" \
        -type f \
        -name "*.pt" \
        | wc -l)

fi


if [ -d "$CD_ROOT" ]; then

    CD_COUNT=$(find "$CD_ROOT" \
        -type f \
        -name "*.pt" \
        | wc -l)

fi


echo "HealthBench Hard:       $HB_COUNT / 188"
echo "HealthBench validation: $VAL_COUNT / 123"
echo "ClinDet external:       $CD_COUNT / 94"
echo


if [ "$HB_COUNT" -ne 188 ]; then

    echo "ERROR:"
    echo "HealthBench Hard cache incomplete."
    exit 1

fi


if [ "$CD_COUNT" -ne 94 ]; then

    echo "ERROR:"
    echo "ClinDet cache incomplete."
    exit 1

fi


if [ "$VAL_COUNT" -eq 0 ]; then

    echo "NOTE:"
    echo "No HealthBench validation cache."
    echo "AUROC/AUPRC will be computed,"
    echo "but threshold-dependent metrics"
    echo "will not be reported."

elif [ "$VAL_COUNT" -ne 123 ]; then

    echo "ERROR:"
    echo "Validation cache is partially complete:"
    echo "$VAL_COUNT / 123"
    exit 1

else

    echo "Validation cache complete."
    echo "EigenScore threshold will be selected"
    echo "on HealthBench validation only."

fi

echo


# ============================================================
# BRIDGE FILE CHECK
# ============================================================

echo "============================================================"
echo "CHECKING BRIDGE PREDICTIONS"
echo "============================================================"


HB_BRIDGE="$BRIDGE_ROOT/$MODEL/healthbench_test_predictions.csv"

CD_BRIDGE="$BRIDGE_ROOT/$MODEL/clindet_external_predictions.csv"


if [ ! -f "$HB_BRIDGE" ]; then

    echo "ERROR:"
    echo "Missing:"
    echo "$HB_BRIDGE"
    exit 1

fi


if [ ! -f "$CD_BRIDGE" ]; then

    echo "ERROR:"
    echo "Missing:"
    echo "$CD_BRIDGE"
    exit 1

fi


echo "BRIDGE prediction files: PASS"
echo


# ============================================================
# LINGSHU CHECK
# ============================================================

if [ "$MODEL" = "lingshu" ]; then

    echo "============================================================"
    echo "CHECKING LINGSHU ARCHITECTURE PATH"
    echo "============================================================"

    if ! grep -q \
        'Qwen2_5_VLForConditionalGeneration' \
        "$SCRIPT"; then

        echo "ERROR:"
        echo "Qwen2.5-VL handling missing."
        exit 1

    fi

    if ! grep -q \
        'qwen2_5_vl' \
        "$SCRIPT"; then

        echo "ERROR:"
        echo "Lingshu model-kind handling missing."
        exit 1

    fi

    echo "Lingshu architecture check: PASS"
    echo

fi


# ============================================================
# OPENBIOLLM CHECK
# ============================================================

if [ "$MODEL" = "openbiollm" ]; then

    echo "============================================================"
    echo "CHECKING OPENBIOLLM PROMPT PATH"
    echo "============================================================"

    if ! grep -q \
        '<|begin_of_text|>' \
        "$SCRIPT"; then

        echo "ERROR:"
        echo "Llama-3 prompt formatting missing."
        exit 1

    fi

    if ! grep -q \
        '<|start_header_id|>' \
        "$SCRIPT"; then

        echo "ERROR:"
        echo "Llama-3 header handling missing."
        exit 1

    fi

    if ! grep -q \
        '<|eot_id|>' \
        "$SCRIPT"; then

        echo "ERROR:"
        echo "Llama-3 EOT handling missing."
        exit 1

    fi

    echo "OpenBioLLM prompt reconstruction: PASS"
    echo

fi


# ============================================================
# RUN
# ============================================================

echo "============================================================"
echo "STARTING EIGENSCORE"
echo "============================================================"
echo


set +e

python -u "$SCRIPT" \
    --models "$MODEL" \
    --generation-root "$GENERATION_ROOT" \
    --bridge-root "$BRIDGE_ROOT" \
    --output-root "$OUTPUT_ROOT"

EXIT_CODE=$?

set -e


echo
echo "============================================================"
echo "EIGENSCORE PROCESS FINISHED"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo


if [ "$EXIT_CODE" -ne 0 ]; then

    echo "============================================================"
    echo "EIGENSCORE FAILED"
    echo "============================================================"
    echo
    echo "Model: $MODEL"
    echo
    echo "stdout:"
    echo "$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
    echo
    echo "stderr:"
    echo "$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
    echo

    exit "$EXIT_CODE"

fi


# ============================================================
# VERIFY OUTPUTS
# ============================================================

MODEL_OUTPUT="$OUTPUT_ROOT/$MODEL"


REQUIRED_FILES=(
    "healthbench_hard_predictions.csv"
    "clindet_external_predictions.csv"
    "metadata.json"
)


for FILE in "${REQUIRED_FILES[@]}"; do

    if [ ! -f "$MODEL_OUTPUT/$FILE" ]; then

        echo "ERROR:"
        echo "Missing output:"
        echo "$MODEL_OUTPUT/$FILE"

        exit 1

    fi

done


if [ "$VAL_COUNT" -eq 123 ]; then

    if [ ! -f "$MODEL_OUTPUT/healthbench_validation_predictions.csv" ]; then

        echo "ERROR:"
        echo "Validation predictions missing."
        exit 1

    fi

    if [ ! -f "$MODEL_OUTPUT/threshold_validation.csv" ]; then

        echo "ERROR:"
        echo "Validation threshold file missing."
        exit 1

    fi

fi


echo "Per-model outputs: PASS"
echo


# ============================================================
# PRINT MODEL RESULTS
# ============================================================

echo "============================================================"
echo "MODEL RESULTS"
echo "============================================================"


python - <<PY

import pandas as pd

summary_path = (
    "$OUTPUT_ROOT/summary.csv"
)

comparison_path = (
    "$OUTPUT_ROOT/paired_comparisons.csv"
)


summary = pd.read_csv(
    summary_path
)

summary = summary[
    summary["model"]
    == "$MODEL"
]


print()
print("EIGENSCORE SUMMARY")
print()

print(
    summary.to_string(
        index=False
    )
)


comparison = pd.read_csv(
    comparison_path
)

comparison = comparison[
    comparison["model"]
    == "$MODEL"
]


print()
print("BRIDGE VS EIGENSCORE")
print()

print(
    comparison.to_string(
        index=False
    )
)

PY


# ============================================================
# COMPLETE
# ============================================================

echo
echo "============================================================"
echo "EIGENSCORE COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo "Model:    $MODEL"
echo
echo "Model outputs:"
echo "$MODEL_OUTPUT"
echo
echo "Global summary:"
echo "$OUTPUT_ROOT/summary.csv"
echo
echo "Paired comparisons:"
echo "$OUTPUT_ROOT/paired_comparisons.csv"
echo "============================================================"