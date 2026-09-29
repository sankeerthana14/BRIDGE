#!/bin/bash

#SBATCH --job-name=neural_text
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

SCRIPT="$REPO_ROOT/scripts/neural_text_classifier.py"

HB_MANIFEST="$REPO_ROOT/data/manifests/healthbench_evidence_sufficiency_v1.csv"

CLINDET_MANIFEST="$REPO_ROOT/data/manifests/clindet_evidence_sufficiency_v1.csv"

BRIDGE_ROOT="$REPO_ROOT/outputs/probes"

OUTPUT_ROOT="$REPO_ROOT/outputs/baselines/neural_text_classifier"

LOG_DIR="$REPO_ROOT/logs"


mkdir -p "$LOG_DIR"

mkdir -p "$OUTPUT_ROOT"

cd "$REPO_ROOT"


# ============================================================
# INFO
# ============================================================

echo "============================================================"
echo "BRIDGE SUPERVISED NEURAL TEXT CLASSIFIER"
echo "============================================================"

echo "Start:       $(date)"
echo "Host:        $(hostname)"
echo "Job ID:      ${SLURM_JOB_ID:-N/A}"

echo

echo "Encoder:     roberta-base"
echo "Seeds:       42 43 44 45 46"

echo

echo "IMPORTANT:"
echo "This baseline is text-only."
echo "No BioMistral/OpenBioLLM/UltraMedical/MedGemma/Lingshu"
echo "model is loaded."

echo

echo "One neural classifier is trained for each seed."
echo "The same predictions are paired against all five BRIDGE probes."

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

print(
    "Environment check: PASS"
)

PY

echo


# ============================================================
# SCRIPT
# ============================================================

if [ ! -f "$SCRIPT" ]; then

    echo "ERROR:"
    echo "Missing:"
    echo "$SCRIPT"

    exit 1

fi


python -m py_compile "$SCRIPT"

echo "Python syntax: PASS"

echo


# ============================================================
# MANIFESTS
# ============================================================

if [ ! -f "$HB_MANIFEST" ]; then

    echo "ERROR:"
    echo "Missing HealthBench manifest:"
    echo "$HB_MANIFEST"

    exit 1

fi


if [ ! -f "$CLINDET_MANIFEST" ]; then

    echo "ERROR:"
    echo "Missing ClinDet manifest:"
    echo "$CLINDET_MANIFEST"

    exit 1

fi


echo "Manifest files: PASS"

echo


# ============================================================
# BRIDGE CHECK
# ============================================================

MODELS=(
    biomistral
    openbiollm
    ultramedical
    medgemma
    lingshu
)


for MODEL in "${MODELS[@]}"; do

    HB_FILE="$BRIDGE_ROOT/$MODEL/healthbench_test_predictions.csv"

    CD_FILE="$BRIDGE_ROOT/$MODEL/clindet_external_predictions.csv"


    if [ ! -f "$HB_FILE" ]; then

        echo "ERROR:"
        echo "Missing $HB_FILE"

        exit 1

    fi


    if [ ! -f "$CD_FILE" ]; then

        echo "ERROR:"
        echo "Missing $CD_FILE"

        exit 1

    fi

done


echo "BRIDGE prediction files: PASS"

echo


# ============================================================
# RUN
# ============================================================

echo "============================================================"
echo "STARTING NEURAL TEXT BASELINE"
echo "============================================================"

echo


set +e


python -u "$SCRIPT" \
    --healthbench-manifest "$HB_MANIFEST" \
    --clindet-manifest "$CLINDET_MANIFEST" \
    --bridge-root "$BRIDGE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --encoder "roberta-base"


EXIT_CODE=$?


set -e


echo

echo "Process exit code: $EXIT_CODE"


if [ "$EXIT_CODE" -ne 0 ]; then

    echo "NEURAL TEXT CLASSIFIER FAILED"

    exit "$EXIT_CODE"

fi


# ============================================================
# VERIFY OUTPUTS
# ============================================================

echo

echo "============================================================"
echo "OUTPUT VERIFICATION"
echo "============================================================"


REQUIRED=(
    "per_seed_metrics.csv"
    "five_seed_summary.csv"
    "primary_seed_summary.csv"
    "paired_comparisons.csv"
    "metadata.json"
)


for FILE in "${REQUIRED[@]}"; do

    if [ ! -f "$OUTPUT_ROOT/$FILE" ]; then

        echo "ERROR:"
        echo "Missing:"
        echo "$OUTPUT_ROOT/$FILE"

        exit 1

    fi

    echo "PASS: $FILE"

done


# ============================================================
# CHECK ALL FIVE SEEDS
# ============================================================

for SEED in 42 43 44 45 46; do

    SEED_DIR="$OUTPUT_ROOT/seed_$SEED"

    REQUIRED_SEED=(
        "training_history.csv"
        "healthbench_validation_predictions.csv"
        "healthbench_hard_predictions.csv"
        "clindet_external_predictions.csv"
        "threshold_validation.csv"
        "best_model.pt"
    )


    for FILE in "${REQUIRED_SEED[@]}"; do

        if [ ! -f "$SEED_DIR/$FILE" ]; then

            echo "ERROR:"
            echo "Seed $SEED missing:"
            echo "$FILE"

            exit 1

        fi

    done


    echo "Seed $SEED outputs: PASS"

done


# ============================================================
# PRINT RESULTS
# ============================================================

echo

echo "============================================================"
echo "NEURAL TEXT RESULTS"
echo "============================================================"


python - <<PY

import pandas as pd


root = "$OUTPUT_ROOT"


print()

print(
    "FIVE-SEED SUMMARY"
)

print()

df = pd.read_csv(
    f"{root}/five_seed_summary.csv"
)

print(
    df.to_string(
        index=False
    )
)


print()

print(
    "PRIMARY SEED 42"
)

print()

df = pd.read_csv(
    f"{root}/primary_seed_summary.csv"
)

print(
    df.to_string(
        index=False
    )
)


print()

print(
    "BRIDGE VS NEURAL TEXT CLASSIFIER"
)

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
echo "NEURAL TEXT CLASSIFIER COMPLETE"
echo "============================================================"

echo "End:    $(date)"

echo "Output: $OUTPUT_ROOT"

echo "============================================================"