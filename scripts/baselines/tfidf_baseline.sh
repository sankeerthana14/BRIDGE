#!/bin/bash

#SBATCH --job-name=tfidf_baseline
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

SCRIPT="$REPO_ROOT/scripts/tfidf_baseline.py"

HB_MANIFEST="$REPO_ROOT/data/manifests/healthbench_evidence_sufficiency_v1.csv"

CLINDET_MANIFEST="$REPO_ROOT/data/manifests/clindet_evidence_sufficiency_v1.csv"

BRIDGE_ROOT="$REPO_ROOT/outputs/probes"

OUTPUT_ROOT="$REPO_ROOT/outputs/baselines/tfidf"

LOG_DIR="$REPO_ROOT/logs"


mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_ROOT"

cd "$REPO_ROOT"


# ============================================================
# INFO
# ============================================================

echo "============================================================"
echo "BRIDGE TF-IDF BASELINE"
echo "============================================================"
echo "Start time:       $(date)"
echo "Host:             $(hostname)"
echo "SLURM job ID:     ${SLURM_JOB_ID:-N/A}"
echo
echo "Script:           $SCRIPT"
echo "HealthBench:      $HB_MANIFEST"
echo "ClinDet:          $CLINDET_MANIFEST"
echo "BRIDGE outputs:   $BRIDGE_ROOT"
echo "Output:           $OUTPUT_ROOT"
echo
echo "NOTE:"
echo "TF-IDF is architecture-independent."
echo "Only one run is required for all five models."
echo "============================================================"
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
import joblib

print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("SciPy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("joblib:", joblib.__version__)
print("Environment check: PASS")

PY

echo


# ============================================================
# FILE CHECKS
# ============================================================

if [ ! -f "$SCRIPT" ]; then

    echo "ERROR: missing script:"
    echo "$SCRIPT"
    exit 1

fi


if [ ! -f "$HB_MANIFEST" ]; then

    echo "ERROR: missing HealthBench manifest."
    exit 1

fi


if [ ! -f "$CLINDET_MANIFEST" ]; then

    echo "ERROR: missing ClinDet manifest."
    exit 1

fi


python -m py_compile "$SCRIPT"

echo "Syntax check: PASS"

echo


# ============================================================
# BRIDGE PREDICTION CHECK
# ============================================================

MODELS=(
    "biomistral"
    "openbiollm"
    "ultramedical"
    "medgemma"
    "lingshu"
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
echo "STARTING TF-IDF"
echo "============================================================"

echo


python -u "$SCRIPT" \
    --healthbench-manifest "$HB_MANIFEST" \
    --clindet-manifest "$CLINDET_MANIFEST" \
    --bridge-root "$BRIDGE_ROOT" \
    --output-root "$OUTPUT_ROOT"


# ============================================================
# VERIFY
# ============================================================

echo
echo "============================================================"
echo "VERIFYING OUTPUTS"
echo "============================================================"


REQUIRED=(
    "tfidf_vectorizer.joblib"
    "tfidf_classifier.joblib"
    "threshold_validation.csv"
    "healthbench_hard_predictions.csv"
    "clindet_external_predictions.csv"
    "summary.csv"
    "paired_comparisons.csv"
    "metadata.json"
)


for FILE in "${REQUIRED[@]}"; do

    if [ ! -f "$OUTPUT_ROOT/$FILE" ]; then

        echo "ERROR:"
        echo "Missing $OUTPUT_ROOT/$FILE"

        exit 1

    fi

    echo "OK: $FILE"

done


# ============================================================
# PRINT RESULTS
# ============================================================

echo
echo "============================================================"
echo "TF-IDF RESULTS"
echo "============================================================"


python - <<PY

import pandas as pd

root = "$OUTPUT_ROOT"

print()
print("TF-IDF SUMMARY")
print()

summary = pd.read_csv(
    f"{root}/summary.csv"
)

print(
    summary.to_string(
        index=False
    )
)

print()
print("BRIDGE VS TF-IDF")
print()

paired = pd.read_csv(
    f"{root}/paired_comparisons.csv"
)

print(
    paired.to_string(
        index=False
    )
)

PY


echo
echo "============================================================"
echo "TF-IDF BASELINE COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo
echo "Results:"
echo "$OUTPUT_ROOT/summary.csv"
echo
echo "Paired comparisons:"
echo "$OUTPUT_ROOT/paired_comparisons.csv"
echo "============================================================"