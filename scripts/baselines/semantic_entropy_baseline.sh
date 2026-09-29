#!/bin/bash

#SBATCH --job-name=semantic_entropy
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

SCRIPT="$REPO_ROOT/scripts/semantic_entropy_baseline.py"

GENERATION_ROOT="$REPO_ROOT/cache/baseline_generations"

BRIDGE_ROOT="$REPO_ROOT/outputs/probes"

OUTPUT_ROOT="$REPO_ROOT/outputs/baselines/semantic_entropy"

LOG_DIR="$REPO_ROOT/logs"


mkdir -p "$LOG_DIR"

mkdir -p "$OUTPUT_ROOT"

cd "$REPO_ROOT"


# ============================================================
# MODEL ARGUMENT
# ============================================================

if [ "$#" -ne 1 ]; then

    echo "ERROR:"
    echo "Supply exactly one model."
    echo
    echo "Usage:"
    echo "sbatch slurm/semantic_entropy.sh MODEL"
    echo
    echo "Allowed models:"
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
# JOB INFO
# ============================================================

echo "============================================================"
echo "BRIDGE COMBINED SEMANTIC ENTROPY"
echo "============================================================"

echo "Start time:       $(date)"
echo "Host:             $(hostname)"
echo "SLURM job ID:     ${SLURM_JOB_ID:-N/A}"
echo "Model:            $MODEL"

echo

echo "Repository:       $REPO_ROOT"
echo "Script:           $SCRIPT"
echo "Generation cache: $GENERATION_ROOT"
echo "BRIDGE outputs:   $BRIDGE_ROOT"
echo "Output root:      $OUTPUT_ROOT"

echo

echo "Pipeline:"
echo "1. Load evaluated medical model"
echo "2. Score existing cached generated token IDs"
echo "3. Save likelihood side-cache"
echo "4. Unload medical model"
echo "5. Load DeBERTa-v2-xlarge-MNLI"
echo "6. Compute Semantic Entropy"
echo "7. Tune threshold on validation"
echo "8. Evaluate Hard + ClinDet"
echo "9. Compare with BRIDGE"

echo

echo "NO GENERATIONS ARE RECREATED."

echo "============================================================"
echo


# ============================================================
# GPU CHECK
# ============================================================

echo "============================================================"
echo "GPU"
echo "============================================================"

nvidia-smi

echo


# ============================================================
# PYTHON ENVIRONMENT
# ============================================================

echo "============================================================"
echo "PYTHON ENVIRONMENT"
echo "============================================================"

which python

python --version


python - <<'PY'

import sys

import numpy
import pandas
import scipy
import sklearn
import torch
import transformers


print("Python:", sys.version)

print("PyTorch:", torch.__version__)

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
    "scikit-learn:",
    sklearn.__version__
)

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
    "Environment check: PASS"
)

PY

echo


# ============================================================
# SCRIPT CHECK
# ============================================================

echo "============================================================"
echo "SCRIPT CHECK"
echo "============================================================"


if [ ! -f "$SCRIPT" ]; then

    echo "ERROR:"
    echo "Missing script:"
    echo "$SCRIPT"

    exit 1

fi


python -m py_compile "$SCRIPT"


echo "Python syntax: PASS"

echo


# ============================================================
# MODEL DIRECTORY
# ============================================================

MODEL_DIR="$REPO_ROOT/models/$MODEL"


if [ ! -d "$MODEL_DIR" ]; then

    echo "ERROR:"
    echo "Missing model directory:"
    echo "$MODEL_DIR"

    exit 1

fi


echo "Model directory: PASS"

echo


# ============================================================
# GENERATION CACHE
# ============================================================

echo "============================================================"
echo "GENERATION CACHE CHECK"
echo "============================================================"


VAL_DIR="$GENERATION_ROOT/$MODEL/healthbench/validation"

HB_DIR="$GENERATION_ROOT/$MODEL/healthbench/test"

CD_DIR="$GENERATION_ROOT/$MODEL/clindet/external_test"


VAL_COUNT=0

HB_COUNT=0

CD_COUNT=0


if [ -d "$VAL_DIR" ]; then

    VAL_COUNT=$(find "$VAL_DIR" \
        -type f \
        -name "*.pt" \
        | wc -l)

fi


if [ -d "$HB_DIR" ]; then

    HB_COUNT=$(find "$HB_DIR" \
        -type f \
        -name "*.pt" \
        | wc -l)

fi


if [ -d "$CD_DIR" ]; then

    CD_COUNT=$(find "$CD_DIR" \
        -type f \
        -name "*.pt" \
        | wc -l)

fi


echo "HealthBench validation: $VAL_COUNT / 123"

echo "HealthBench Hard:       $HB_COUNT / 188"

echo "ClinDet external:       $CD_COUNT / 94"

echo


if [ "$VAL_COUNT" -ne 123 ]; then

    echo "ERROR:"
    echo "Validation cache incomplete."

    exit 1

fi


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


echo "Generation cache counts: PASS"

echo


# ============================================================
# DEEP GENERATION CACHE CHECK
# ============================================================

echo "============================================================"
echo "DEEP CACHE CHECK"
echo "============================================================"


python - <<PY

from pathlib import Path

import torch


roots = {
    "validation":
        Path("$VAL_DIR"),

    "healthbench_hard":
        Path("$HB_DIR"),

    "clindet_external":
        Path("$CD_DIR"),
}


expected = {
    "validation":
        123,

    "healthbench_hard":
        188,

    "clindet_external":
        94,
}


for name, root in roots.items():

    files = sorted(
        root.glob("*.pt")
    )


    assert (
        len(files)
        ==
        expected[name]
    ), (
        name,
        len(files),
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
            "samples",
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


        samples = obj[
            "samples"
        ]


        if len(samples) != 10:

            raise RuntimeError(
                f"{path}: "
                f"expected 10 samples, "
                f"got {len(samples)}"
            )


        for i, sample in enumerate(
            samples
        ):

            if (
                "generated_token_ids"
                not in sample
            ):

                raise RuntimeError(
                    f"{path}: "
                    f"sample {i} missing "
                    "generated_token_ids"
                )


            if (
                len(
                    sample[
                        "generated_token_ids"
                    ]
                )
                ==
                0
            ):

                raise RuntimeError(
                    f"{path}: "
                    f"sample {i} empty"
                )


    print(
        name,
        "PASS:",
        len(files)
    )


print(
    "Deep generation cache validation: PASS"
)

PY

echo


# ============================================================
# BRIDGE FILES
# ============================================================

echo "============================================================"
echo "BRIDGE PREDICTION CHECK"
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


echo "BRIDGE files: PASS"

echo


# ============================================================
# ARCHITECTURE CHECKS
# ============================================================

echo "============================================================"
echo "ARCHITECTURE CHECK"
echo "============================================================"


if [ "$MODEL" = "lingshu" ]; then

    if ! grep -q \
        "Qwen2_5_VLForConditionalGeneration" \
        "$SCRIPT"; then

        echo "ERROR:"
        echo "Lingshu Qwen2.5-VL handling missing."

        exit 1

    fi

    echo "Lingshu Qwen2.5-VL handling: PASS"

fi


if [ "$MODEL" = "medgemma" ]; then

    if ! grep -q \
        "Gemma3ForConditionalGeneration" \
        "$SCRIPT"; then

        echo "ERROR:"
        echo "MedGemma Gemma3 handling missing."

        exit 1

    fi

    echo "MedGemma Gemma3 handling: PASS"

fi


if [ "$MODEL" = "openbiollm" ]; then

    if ! grep -q \
        "<|begin_of_text|>" \
        "$SCRIPT"; then

        echo "ERROR:"
        echo "OpenBioLLM explicit "
        echo "Llama-3 formatting missing."

        exit 1

    fi

    echo "OpenBioLLM Llama-3 formatting: PASS"

fi


echo


# ============================================================
# RUN
# ============================================================

echo "============================================================"
echo "STARTING SEMANTIC ENTROPY"
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

echo "============================================================"
echo "PROCESS FINISHED"
echo "============================================================"

echo "Exit code: $EXIT_CODE"

echo


if [ "$EXIT_CODE" -ne 0 ]; then

    echo "============================================================"
    echo "SEMANTIC ENTROPY FAILED"
    echo "============================================================"

    echo

    echo "Model: $MODEL"

    echo

    echo "stdout:"
    echo "$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"

    echo

    echo "stderr:"
    echo "$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

    exit "$EXIT_CODE"

fi


# ============================================================
# OUTPUT CHECK
# ============================================================

echo "============================================================"
echo "VERIFYING OUTPUTS"
echo "============================================================"


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


for FILE in "${REQUIRED_FILES[@]}"; do

    if [ ! -f "$MODEL_OUTPUT/$FILE" ]; then

        echo "ERROR:"
        echo "Missing:"
        echo "$MODEL_OUTPUT/$FILE"

        exit 1

    fi

    echo "PASS: $FILE"

done


echo


# ============================================================
# LIKELIHOOD CACHE VERIFICATION
# ============================================================

echo "============================================================"
echo "LIKELIHOOD CACHE VERIFICATION"
echo "============================================================"


for SPEC in \
    "healthbench_validation:123" \
    "healthbench_hard:188" \
    "clindet_external:94"
do

    NAME="${SPEC%%:*}"

    EXPECTED="${SPEC##*:}"

    DIR="$MODEL_OUTPUT/likelihood_cache/$NAME"

    COUNT=$(find "$DIR" \
        -type f \
        -name "*.pt" \
        | wc -l)


    echo "$NAME: $COUNT / $EXPECTED"


    if [ "$COUNT" -ne "$EXPECTED" ]; then

        echo "ERROR:"
        echo "Likelihood cache incomplete."

        exit 1

    fi

done


echo "Likelihood cache verification: PASS"

echo


# ============================================================
# PRINT RESULTS
# ============================================================

echo "============================================================"
echo "SEMANTIC ENTROPY RESULTS"
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
    "BRIDGE VS SEMANTIC ENTROPY"
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
echo "SEMANTIC ENTROPY COMPLETE"
echo "============================================================"

echo "End time: $(date)"

echo "Model:    $MODEL"

echo

echo "Results:"
echo "$MODEL_OUTPUT"

echo

echo "============================================================"