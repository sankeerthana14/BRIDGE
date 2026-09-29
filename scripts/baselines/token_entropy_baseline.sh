#!/bin/bash

#SBATCH --job-name=token_entropy
#SBATCH --gres=gpu:1
#SBATCH --partition=RTXA6Kq
#SBATCH --nodelist=node09
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

SCRIPT="$REPO_ROOT/scripts/token_entropy_baseline.py"

GENERATION_ROOT="$REPO_ROOT/cache/baseline_generations"

BRIDGE_ROOT="$REPO_ROOT/outputs/probes"

OUTPUT_ROOT="$REPO_ROOT/outputs/baselines/token_entropy"

LOG_DIR="$REPO_ROOT/logs"


mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_ROOT"

cd "$REPO_ROOT"


# ============================================================
# ARGUMENT
# ============================================================

if [ "$#" -ne 1 ]; then

    echo "Usage:"
    echo "sbatch slurm/token_entropy.sh MODEL"
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

        echo "ERROR: unknown model $MODEL"

        exit 1
        ;;

esac


# ============================================================
# INFO
# ============================================================

echo "============================================================"
echo "BRIDGE TOKEN ENTROPY BASELINE"
echo "============================================================"

echo "Start:        $(date)"
echo "Host:         $(hostname)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Model:        $MODEL"

echo
echo "Generation:   existing cached generations"
echo "Regeneration: NONE"

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

import torch
import transformers
import numpy
import pandas
import sklearn
import scipy

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("sklearn:", sklearn.__version__)
print("SciPy:", scipy.__version__)

print(
    "CUDA:",
    torch.cuda.is_available()
)

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA unavailable."
    )

from transformers import (
    Gemma3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

print("Gemma3 import: PASS")
print("Qwen2.5-VL import: PASS")
print("Environment: PASS")

PY


# ============================================================
# SCRIPT
# ============================================================

if [ ! -f "$SCRIPT" ]; then

    echo "ERROR: missing $SCRIPT"

    exit 1

fi


python -m py_compile "$SCRIPT"

echo "Python syntax: PASS"


# ============================================================
# CACHE COUNTS
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
echo "CACHE CHECK"
echo "============================================================"

echo "Validation: $VAL_COUNT / 123"
echo "HB Hard:    $HB_COUNT / 188"
echo "ClinDet:    $CD_COUNT / 94"


if [ "$VAL_COUNT" -ne 123 ]; then
    echo "ERROR: validation incomplete"
    exit 1
fi


if [ "$HB_COUNT" -ne 188 ]; then
    echo "ERROR: HealthBench incomplete"
    exit 1
fi


if [ "$CD_COUNT" -ne 94 ]; then
    echo "ERROR: ClinDet incomplete"
    exit 1
fi


echo "Cache count check: PASS"


# ============================================================
# DEEP CACHE CHECK
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
        root.glob("*.pt")
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
            "samples",
        }

        missing = (
            required
            - set(obj.keys())
        )

        if missing:

            raise RuntimeError(
                f"{path}: missing {missing}"
            )

        if len(
            obj["samples"]
        ) != 10:

            raise RuntimeError(
                f"{path}: not 10 samples"
            )

        for i, sample in enumerate(
            obj["samples"]
        ):

            ids = sample.get(
                "generated_token_ids",
                None,
            )

            if not isinstance(
                ids,
                (list, tuple),
            ):

                raise RuntimeError(
                    f"{path}: sample {i} "
                    "missing generated_token_ids"
                )

            if len(ids) == 0:

                raise RuntimeError(
                    f"{path}: sample {i} empty"
                )


print(
    "Deep cache validation: PASS"
)

PY


# ============================================================
# BRIDGE FILES
# ============================================================

HB_BRIDGE="$BRIDGE_ROOT/$MODEL/healthbench_test_predictions.csv"

CD_BRIDGE="$BRIDGE_ROOT/$MODEL/clindet_external_predictions.csv"


if [ ! -f "$HB_BRIDGE" ]; then

    echo "ERROR: missing $HB_BRIDGE"
    exit 1

fi


if [ ! -f "$CD_BRIDGE" ]; then

    echo "ERROR: missing $CD_BRIDGE"
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

    echo "OpenBioLLM prompt formatting: PASS"

fi


# ============================================================
# RUN
# ============================================================

echo
echo "============================================================"
echo "STARTING TOKEN ENTROPY"
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

    echo "TOKEN ENTROPY FAILED"

    exit "$EXIT_CODE"

fi


# ============================================================
# OUTPUT VERIFICATION
# ============================================================

MODEL_OUTPUT="$OUTPUT_ROOT/$MODEL"


REQUIRED=(
    "healthbench_validation_predictions.csv"
    "healthbench_hard_predictions.csv"
    "clindet_external_predictions.csv"
    "threshold_validation.csv"
    "summary.csv"
    "paired_comparisons.csv"
    "metadata.json"
)


for FILE in "${REQUIRED[@]}"; do

    if [ ! -f "$MODEL_OUTPUT/$FILE" ]; then

        echo "ERROR:"
        echo "Missing $MODEL_OUTPUT/$FILE"

        exit 1

    fi

    echo "PASS: $FILE"

done


# ============================================================
# PRINT RESULTS
# ============================================================

echo
echo "============================================================"
echo "TOKEN ENTROPY RESULTS"
echo "============================================================"


python - <<PY

import pandas as pd


root = "$MODEL_OUTPUT"


print()
print("SUMMARY")
print()

df = pd.read_csv(
    f"{root}/summary.csv"
)

print(
    df.to_string(
        index=False
    )
)


print()
print("BRIDGE VS TOKEN ENTROPY")
print()

df = pd.read_csv(
    f"{root}/paired_comparisons.csv"
)

print(
    df.to_string(
        index=False
    )
)

PY


echo
echo "============================================================"
echo "TOKEN ENTROPY COMPLETE"
echo "============================================================"

echo "End:   $(date)"
echo "Model: $MODEL"
echo "Output: $MODEL_OUTPUT"
echo "============================================================"