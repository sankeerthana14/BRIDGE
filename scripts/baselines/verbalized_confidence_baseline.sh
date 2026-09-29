#!/bin/bash

#SBATCH --job-name=vc_constrained
#SBATCH --gres=gpu:1
#SBATCH --partition=RTXA6Kq
#SBATCH --nodelist=node15
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

SCRIPT="$REPO_ROOT/scripts/verbalized_confidence_baseline.py"

GENERATION_ROOT="$REPO_ROOT/cache/baseline_generations"

BRIDGE_ROOT="$REPO_ROOT/outputs/probes"

OUTPUT_ROOT="$REPO_ROOT/outputs/baselines/verbalized_confidence_v2"

LOG_DIR="$REPO_ROOT/logs"


mkdir -p "$LOG_DIR"

mkdir -p "$OUTPUT_ROOT"

cd "$REPO_ROOT"


# ============================================================
# MODEL
# ============================================================

if [ "$#" -ne 1 ]; then

    echo "Usage:"
    echo "sbatch slurm/verbalized_confidence.sh MODEL"

    echo

    echo "MODEL:"
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
# INFO
# ============================================================

echo "============================================================"
echo "BRIDGE CONSTRAINED VERBALIZED CONFIDENCE"
echo "============================================================"

echo "Start:    $(date)"
echo "Host:     $(hostname)"
echo "Job ID:   ${SLURM_JOB_ID:-N/A}"
echo "Model:    $MODEL"

echo

echo "Candidate scores:"
echo "0 10 20 30 40 50 60 70 80 90 100"

echo

echo "No free-form generation."
echo "Each candidate is scored directly under the model."

echo "============================================================"
echo


# ============================================================
# GPU
# ============================================================

nvidia-smi

echo


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

which python

python --version


python - <<'PY'

import numpy
import pandas
import scipy
import sklearn
import torch
import transformers


print(
    "PyTorch:",
    torch.__version__
)

print(
    "Transformers:",
    transformers.__version__
)

print(
    "NumPy:",
    numpy.__version__
)

print(
    "pandas:",
    pandas.__version__
)

print(
    "SciPy:",
    scipy.__version__
)

print(
    "sklearn:",
    sklearn.__version__
)


print(
    "CUDA:",
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
    Gemma3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)


print(
    "Gemma3 import: PASS"
)

print(
    "Qwen2.5-VL import: PASS"
)

print(
    "Environment: PASS"
)

PY


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


echo "Python syntax: PASS"


# ============================================================
# CACHE CHECK
# ============================================================

VAL_DIR="$GENERATION_ROOT/$MODEL/healthbench/validation"

HB_DIR="$GENERATION_ROOT/$MODEL/healthbench/test"

CD_DIR="$GENERATION_ROOT/$MODEL/clindet/external_test"


VAL_COUNT=$(find "$VAL_DIR" \
    -type f \
    -name "*.pt" \
    | wc -l)


HB_COUNT=$(find "$HB_DIR" \
    -type f \
    -name "*.pt" \
    | wc -l)


CD_COUNT=$(find "$CD_DIR" \
    -type f \
    -name "*.pt" \
    | wc -l)


echo

echo "============================================================"
echo "SOURCE CACHE CHECK"
echo "============================================================"


echo "Validation: $VAL_COUNT / 123"

echo "HB Hard:    $HB_COUNT / 188"

echo "ClinDet:    $CD_COUNT / 94"


if [ "$VAL_COUNT" -ne 123 ]; then

    echo "ERROR:"
    echo "Validation cache incomplete."

    exit 1

fi


if [ "$HB_COUNT" -ne 188 ]; then

    echo "ERROR:"
    echo "HealthBench cache incomplete."

    exit 1

fi


if [ "$CD_COUNT" -ne 94 ]; then

    echo "ERROR:"
    echo "ClinDet cache incomplete."

    exit 1

fi


echo "Cache counts: PASS"


# ============================================================
# DEEP SOURCE CHECK
# ============================================================

python - <<PY

from pathlib import Path

import torch


specs = [
    (
        Path("$VAL_DIR"),
        123,
    ),

    (
        Path("$HB_DIR"),
        188,
    ),

    (
        Path("$CD_DIR"),
        94,
    ),
]


for root, expected in specs:

    files = sorted(
        root.glob(
            "*.pt"
        )
    )


    if len(files) != expected:

        raise RuntimeError(
            f"{root}: "
            f"{len(files)} != {expected}"
        )


    for path in files:

        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )


        required = {
            "example_id",
            "label",
            "prompt",
        }


        missing = (
            required
            -
            set(
                obj.keys()
            )
        )


        if missing:

            raise RuntimeError(
                f"{path}: "
                f"missing {missing}"
            )


print(
    "Deep source-cache check: PASS"
)

PY


# ============================================================
# BRIDGE CHECK
# ============================================================

HB_BRIDGE="$BRIDGE_ROOT/$MODEL/healthbench_test_predictions.csv"

CD_BRIDGE="$BRIDGE_ROOT/$MODEL/clindet_external_predictions.csv"


if [ ! -f "$HB_BRIDGE" ]; then

    echo "ERROR:"
    echo "Missing $HB_BRIDGE"

    exit 1

fi


if [ ! -f "$CD_BRIDGE" ]; then

    echo "ERROR:"
    echo "Missing $CD_BRIDGE"

    exit 1

fi


echo "BRIDGE files: PASS"


# ============================================================
# ARCHITECTURE CHECK
# ============================================================

if [ "$MODEL" = "lingshu" ]; then

    grep -q \
        "Qwen2_5_VLForConditionalGeneration" \
        "$SCRIPT"

    echo "Lingshu architecture: PASS"

fi


if [ "$MODEL" = "medgemma" ]; then

    grep -q \
        "Gemma3ForConditionalGeneration" \
        "$SCRIPT"

    echo "MedGemma architecture: PASS"

fi


if [ "$MODEL" = "openbiollm" ]; then

    grep -q \
        "<|begin_of_text|>" \
        "$SCRIPT"

    echo "OpenBioLLM formatting: PASS"

fi


# ============================================================
# RUN
# ============================================================

echo

echo "============================================================"
echo "STARTING CONSTRAINED VC"
echo "============================================================"

echo


set +e


python -u "$SCRIPT" \
    --model "$MODEL" \
    --generation-root "$GENERATION_ROOT" \
    --bridge-root "$BRIDGE_ROOT" \
    --output-root "$OUTPUT_ROOT"


EXIT_CODE=$?


set -e


echo

echo "Process exit code: $EXIT_CODE"


if [ "$EXIT_CODE" -ne 0 ]; then

    echo "VERBALIZED CONFIDENCE FAILED"

    exit "$EXIT_CODE"

fi


# ============================================================
# OUTPUT CHECK
# ============================================================

MODEL_OUTPUT="$OUTPUT_ROOT/$MODEL"


REQUIRED_FILES=(
    "healthbench_validation_predictions.csv"
    "healthbench_hard_predictions.csv"
    "clindet_external_predictions.csv"
    "threshold_validation.csv"
    "summary.csv"
    "paired_comparisons.csv"
    "metadata.json"
)


echo

echo "============================================================"
echo "OUTPUT CHECK"
echo "============================================================"


for FILE in "${REQUIRED_FILES[@]}"; do

    if [ ! -f "$MODEL_OUTPUT/$FILE" ]; then

        echo "ERROR:"
        echo "Missing:"
        echo "$MODEL_OUTPUT/$FILE"

        exit 1

    fi

    echo "PASS: $FILE"

done


# ============================================================
# SCORE CACHE COUNTS
# ============================================================

echo

echo "============================================================"
echo "SCORE CACHE CHECK"
echo "============================================================"


for SPEC in \
    "healthbench_validation:123" \
    "healthbench_hard:188" \
    "clindet_external:94"
do

    NAME="${SPEC%%:*}"

    EXPECTED="${SPEC##*:}"

    CACHE_DIR="$MODEL_OUTPUT/score_cache/$NAME"


    COUNT=$(find "$CACHE_DIR" \
        -type f \
        -name "*.pt" \
        | wc -l)


    echo "$NAME: $COUNT / $EXPECTED"


    if [ "$COUNT" -ne "$EXPECTED" ]; then

        echo "ERROR:"
        echo "Score cache incomplete."

        exit 1

    fi

done


echo "Score-cache counts: PASS"


# ============================================================
# PRINT RESULTS
# ============================================================

echo

echo "============================================================"
echo "CONSTRAINED VERBALIZED CONFIDENCE RESULTS"
echo "============================================================"


python - <<PY

import pandas as pd


root = "$MODEL_OUTPUT"


summary = pd.read_csv(
    f"{root}/summary.csv"
)


comparison = pd.read_csv(
    f"{root}/paired_comparisons.csv"
)


print()

print(
    "SUMMARY"
)

print()

print(
    summary.to_string(
        index=False
    )
)


print()

print(
    "BRIDGE VS VERBALIZED CONFIDENCE"
)

print()

print(
    comparison.to_string(
        index=False
    )
)

PY


echo

echo "============================================================"
echo "VERBALIZED CONFIDENCE COMPLETE"
echo "============================================================"

echo "End:    $(date)"

echo "Model:  $MODEL"

echo "Output: $MODEL_OUTPUT"

echo "============================================================"