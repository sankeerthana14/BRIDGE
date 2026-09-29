#!/bin/bash

#SBATCH --job-name=bridge_saplma
#SBATCH --gres=gpu:1
#SBATCH --partition=RTXA6Kq
#SBATCH --nodelist=node16
#SBATCH --output=/export/home2/sati0004/JBHI/BRIDGE/logs/%x_%j.out
#SBATCH --error=/export/home2/sati0004/JBHI/BRIDGE/logs/%x_%j.err

set -euo pipefail


# ============================================================
# Environment
# ============================================================

module purge

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate mech_interp


# ============================================================
# Paths
# ============================================================

REPO_ROOT="$HOME/JBHI/BRIDGE"

SCRIPT="$REPO_ROOT/scripts/saplma_baseline.py"

ACTIVATION_ROOT="$REPO_ROOT/cache/activations"

BRIDGE_ROOT="$REPO_ROOT/outputs/probes"

OUTPUT_ROOT="$REPO_ROOT/outputs/baselines/saplma"

LOG_DIR="$REPO_ROOT/logs"


mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_ROOT"

cd "$REPO_ROOT"


# ============================================================
# Run information
# ============================================================

echo "============================================================"
echo "BRIDGE SAPLMA BASELINE"
echo "============================================================"
echo "Start time:       $(date)"
echo "Host:             $(hostname)"
echo "SLURM job ID:     ${SLURM_JOB_ID:-N/A}"
echo "Repository root:  $REPO_ROOT"
echo "Script:           $SCRIPT"
echo "Activation root:  $ACTIVATION_ROOT"
echo "BRIDGE root:      $BRIDGE_ROOT"
echo "Output root:      $OUTPUT_ROOT"
echo "Log directory:    $LOG_DIR"
echo "============================================================"
echo


# ============================================================
# GPU information
# ============================================================

echo "============================================================"
echo "GPU INFORMATION"
echo "============================================================"

nvidia-smi

echo


# ============================================================
# Python environment
# ============================================================

echo "============================================================"
echo "PYTHON ENVIRONMENT"
echo "============================================================"

echo "Python executable:"
which python

echo

echo "Python version:"
python --version

echo


python - <<'PY'
import sys
import numpy
import pandas
import sklearn
import scipy
import torch

print("Python:", sys.version)
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("SciPy:", scipy.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available inside the SLURM job."
    )

print("GPU:", torch.cuda.get_device_name(0))
print()
print("Environment check: PASS")
PY

echo


# ============================================================
# Check script
# ============================================================

echo "============================================================"
echo "CHECKING SAPLMA SCRIPT"
echo "============================================================"

if [ ! -f "$SCRIPT" ]; then

    echo "ERROR:"
    echo "SAPLMA script not found:"
    echo "$SCRIPT"

    exit 1

fi

echo "Found:"
echo "$SCRIPT"

echo


# ============================================================
# Python syntax check
# ============================================================

echo "Running Python syntax check..."

python -m py_compile "$SCRIPT"

echo "Syntax check: PASS"

echo


# ============================================================
# Check activation cache
# ============================================================

echo "============================================================"
echo "CHECKING ACTIVATION CACHE"
echo "============================================================"

if [ ! -d "$ACTIVATION_ROOT" ]; then

    echo "ERROR:"
    echo "Activation directory not found:"
    echo "$ACTIVATION_ROOT"

    exit 1

fi


NUM_ACTIVATIONS=$(find "$ACTIVATION_ROOT" \
    -type f \
    -name "*.pt" \
    | wc -l)


echo "Activation files found: $NUM_ACTIVATIONS"
echo "Expected:               4485"
echo


if [ "$NUM_ACTIVATIONS" -ne 4485 ]; then

    echo "ERROR:"
    echo "Expected 4485 validated activation files."
    echo "Found $NUM_ACTIVATIONS."

    exit 1

fi

echo "Activation cache: PASS"

echo


# ============================================================
# Check existing BRIDGE probe outputs
# ============================================================

echo "============================================================"
echo "CHECKING EXISTING BRIDGE RESULTS"
echo "============================================================"

if [ ! -d "$BRIDGE_ROOT" ]; then

    echo "ERROR:"
    echo "BRIDGE probe output directory not found:"
    echo "$BRIDGE_ROOT"

    exit 1

fi


if [ ! -f "$BRIDGE_ROOT/summary.csv" ]; then

    echo "ERROR:"
    echo "BRIDGE summary.csv not found:"
    echo "$BRIDGE_ROOT/summary.csv"

    exit 1

fi


MODELS=(
    "biomistral"
    "openbiollm"
    "ultramedical"
    "medgemma"
    "lingshu"
)


for MODEL in "${MODELS[@]}"; do

    if [ ! -f "$BRIDGE_ROOT/$MODEL/healthbench_test_predictions.csv" ]; then

        echo "ERROR:"
        echo "Missing BRIDGE HealthBench predictions for $MODEL"

        exit 1

    fi


    if [ ! -f "$BRIDGE_ROOT/$MODEL/clindet_external_predictions.csv" ]; then

        echo "ERROR:"
        echo "Missing BRIDGE ClinDet predictions for $MODEL"

        exit 1

    fi

done


echo "Existing BRIDGE outputs: PASS"
echo


# ============================================================
# Experiment description
# ============================================================

echo "============================================================"
echo "EXPERIMENT"
echo "============================================================"
echo "Models:"
echo "  biomistral"
echo "  openbiollm"
echo "  ultramedical"
echo "  medgemma"
echo "  lingshu"
echo
echo "SAPLMA-style MLP:"
echo "  hidden_size -> 256 -> 128 -> 64 -> 1"
echo
echo "Seeds:"
echo "  42, 43, 44, 45, 46"
echo
echo "Layer selection:"
echo "  HealthBench validation AUROC"
echo
echo "Threshold selection:"
echo "  HealthBench validation balanced accuracy"
echo
echo "Evaluation:"
echo "  HealthBench Hard"
echo "  ClinDet external test"
echo
echo "Statistics:"
echo "  bootstrap confidence intervals"
echo "  paired bootstrap differences"
echo "  paired randomization tests"
echo "  McNemar exact test"
echo "============================================================"
echo


# ============================================================
# Run SAPLMA experiment
# ============================================================

echo "============================================================"
echo "STARTING SAPLMA BASELINE"
echo "============================================================"
echo


set +e

python -u "$SCRIPT" \
    --models all \
    --activation-root "$ACTIVATION_ROOT" \
    --bridge-root "$BRIDGE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --log-dir "$LOG_DIR"

EXIT_CODE=$?

set -e


echo
echo "============================================================"
echo "SAPLMA PROCESS FINISHED"
echo "============================================================"
echo "Exit code: $EXIT_CODE"
echo


# ============================================================
# Stop if experiment failed
# ============================================================

if [ "$EXIT_CODE" -ne 0 ]; then

    echo "============================================================"
    echo "SAPLMA BASELINE FAILED"
    echo "============================================================"
    echo
    echo "SLURM stdout:"
    echo "$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
    echo
    echo "SLURM stderr:"
    echo "$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
    echo

    exit "$EXIT_CODE"

fi


# ============================================================
# Verify expected outputs
# ============================================================

echo "============================================================"
echo "VERIFYING OUTPUTS"
echo "============================================================"


REQUIRED_FILES=(
    "summary_primary.csv"
    "seed_results.csv"
    "seed_summary.csv"
    "paired_comparisons.csv"
)


for FILE in "${REQUIRED_FILES[@]}"; do

    PATH_TO_FILE="$OUTPUT_ROOT/$FILE"

    if [ ! -f "$PATH_TO_FILE" ]; then

        echo "ERROR:"
        echo "Missing expected output:"
        echo "$PATH_TO_FILE"

        exit 1

    fi

    echo "OK: $FILE"

done

echo


# ============================================================
# Verify per-model SAPLMA artifacts
# ============================================================

for MODEL in "${MODELS[@]}"; do

    MODEL_ROOT="$OUTPUT_ROOT/$MODEL"

    echo "$MODEL"

    REQUIRED_MODEL_FILES=(
        "saplma_primary.pt"
        "primary_metadata.json"
        "healthbench_predictions.csv"
        "clindet_predictions.csv"
        "layer_validation_seed42.csv"
        "threshold_validation_seed42.csv"
    )

    for FILE in "${REQUIRED_MODEL_FILES[@]}"; do

        if [ ! -f "$MODEL_ROOT/$FILE" ]; then

            echo "ERROR:"
            echo "Missing:"
            echo "$MODEL_ROOT/$FILE"

            exit 1

        fi

        echo "  OK  $FILE"

    done

    echo

done


# ============================================================
# Print main results
# ============================================================

echo "============================================================"
echo "PRIMARY SAPLMA RESULTS"
echo "============================================================"

python - <<PY
import pandas as pd

root = "$OUTPUT_ROOT"

primary = pd.read_csv(
    f"{root}/summary_primary.csv"
)

print(primary.to_string(index=False))
PY

echo


echo "============================================================"
echo "BRIDGE VS SAPLMA PAIRED COMPARISONS"
echo "============================================================"

python - <<PY
import pandas as pd

root = "$OUTPUT_ROOT"

df = pd.read_csv(
    f"{root}/paired_comparisons.csv"
)

print(df.to_string(index=False))
PY

echo


echo "============================================================"
echo "SEED STABILITY"
echo "============================================================"

python - <<PY
import pandas as pd

root = "$OUTPUT_ROOT"

df = pd.read_csv(
    f"{root}/seed_summary.csv"
)

print(df.to_string(index=False))
PY

echo


# ============================================================
# Find newest detailed SAPLMA log
# ============================================================

LATEST_LOG=$(find "$LOG_DIR" \
    -maxdepth 1 \
    -type f \
    -name "saplma_baseline_*.log" \
    -printf "%T@ %p\n" \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-)


# ============================================================
# Done
# ============================================================

echo
echo "============================================================"
echo "BRIDGE SAPLMA BASELINE COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo
echo "Primary results:"
echo "$OUTPUT_ROOT/summary_primary.csv"
echo
echo "Seed results:"
echo "$OUTPUT_ROOT/seed_results.csv"
echo
echo "Seed summary:"
echo "$OUTPUT_ROOT/seed_summary.csv"
echo
echo "Paired comparisons:"
echo "$OUTPUT_ROOT/paired_comparisons.csv"
echo
echo "Detailed log:"
echo "${LATEST_LOG:-Not found}"
echo
echo "SLURM stdout:"
echo "$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo
echo "SLURM stderr:"
echo "$LOG_DIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
echo
echo "============================================================"