#!/bin/bash

#SBATCH --job-name=selfcheckgpt
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

SCRIPT="$REPO_ROOT/scripts/selfcheckgpt_baseline.py"

GENERATION_ROOT="$REPO_ROOT/cache/baseline_generations"

BRIDGE_ROOT="$REPO_ROOT/outputs/probes"

OUTPUT_ROOT="$REPO_ROOT/outputs/baselines/selfcheckgpt"

LOG_DIR="$REPO_ROOT/logs"


mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_ROOT"

cd "$REPO_ROOT"


# ============================================================
# MODEL ARGUMENT
# ============================================================

if [ "$#" -ne 1 ]; then

    echo "ERROR:"
    echo "Exactly one model name is required."
    echo
    echo "Usage:"
    echo "sbatch slurm/selfcheckgpt.sh MODEL"
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
echo "BRIDGE SELFCHECKGPT-NLI"
echo "============================================================"
echo "Start time:       $(date)"
echo "Host:             $(hostname)"
echo "SLURM job ID:     ${SLURM_JOB_ID:-N/A}"
echo "Model:            $MODEL"
echo
echo "Repository root:  $REPO_ROOT"
echo "Script:           $SCRIPT"
echo "Generation root:  $GENERATION_ROOT"
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
# ENVIRONMENT VALIDATION
# ============================================================

echo "============================================================"
echo "PYTHON ENVIRONMENT"
echo "============================================================"

which python
python --version

python - <<'PY'

import torch
import transformers
import numpy
import pandas
import sklearn

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA unavailable."
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "selfcheckgpt import: PASS"
)

PY

echo


# ============================================================
# SCRIPT CHECK
# ============================================================

echo "============================================================"
echo "CHECKING SCRIPT"
echo "============================================================"

if [ ! -f "$SCRIPT" ]; then

    echo "ERROR:"
    echo "Script not found:"
    echo "$SCRIPT"

    exit 1

fi


python -m py_compile "$SCRIPT"

echo "Syntax check: PASS"
echo


# ============================================================
# GENERATION CACHE PATHS
# ============================================================

HB_ROOT="$GENERATION_ROOT/$MODEL/healthbench/test"

CD_ROOT="$GENERATION_ROOT/$MODEL/clindet/external_test"

VAL_ROOT="$GENERATION_ROOT/$MODEL/healthbench/validation"


HB_COUNT=0
CD_COUNT=0
VAL_COUNT=0


if [ -d "$HB_ROOT" ]; then

    HB_COUNT=$(find "$HB_ROOT" \
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


if [ -d "$VAL_ROOT" ]; then

    VAL_COUNT=$(find "$VAL_ROOT" \
        -type f \
        -name "*.pt" \
        | wc -l)

fi


# ============================================================
# CACHE CHECK
# ============================================================

echo "============================================================"
echo "CHECKING GENERATION CACHE"
echo "============================================================"

echo "Model:                  $MODEL"
echo "HealthBench Hard:       $HB_COUNT / 188"
echo "ClinDet external:       $CD_COUNT / 94"
echo "HealthBench validation: $VAL_COUNT / 123"
echo


if [ "$HB_COUNT" -ne 188 ]; then

    echo "ERROR:"
    echo "Expected 188 HealthBench Hard files for $MODEL."

    exit 1

fi


if [ "$CD_COUNT" -ne 94 ]; then

    echo "ERROR:"
    echo "Expected 94 ClinDet files for $MODEL."

    exit 1

fi


if [ "$VAL_COUNT" -eq 0 ]; then

    echo "NOTE:"
    echo "No HealthBench validation generations yet."
    echo
    echo "The script will compute:"
    echo "  AUROC"
    echo "  AUPRC"
    echo "  bootstrap confidence intervals"
    echo
    echo "Threshold-dependent metrics will be skipped."
    echo

elif [ "$VAL_COUNT" -ne 123 ]; then

    echo "ERROR:"
    echo "Validation cache exists but is incomplete:"
    echo "$VAL_COUNT / 123"

    exit 1

else

    echo "Validation cache complete."
    echo
    echo "Threshold will be selected using:"
    echo "  HealthBench validation balanced accuracy"
    echo
    echo "Then frozen for:"
    echo "  HealthBench Hard"
    echo "  ClinDet external"
    echo

fi


# ============================================================
# OPTIONAL DEEP CACHE CHECK
# ============================================================

echo "============================================================"
echo "DEEP CACHE CHECK"
echo "============================================================"

python - <<PY

from pathlib import Path
import torch

model = "$MODEL"

roots = [
    Path("$HB_ROOT"),
    Path("$CD_ROOT"),
]

if Path("$VAL_ROOT").exists():

    roots.append(
        Path("$VAL_ROOT")
    )


required = {
    "example_id",
    "label",
    "prompt",
    "greedy",
    "samples",
    "generation_config",
}


total = 0


for root in roots:

    files = sorted(
        root.glob(
            "*.pt"
        )
    )

    for path in files:

        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if not required.issubset(
            obj.keys()
        ):

            raise RuntimeError(
                f"Missing required keys: {path}"
            )


        greedy = str(
            obj[
                "greedy"
            ].get(
                "text",
                "",
            )
        ).strip()


        if not greedy:

            raise RuntimeError(
                f"Empty greedy response: {path}"
            )


        samples = obj[
            "samples"
        ]


        if len(samples) != 10:

            raise RuntimeError(
                f"Expected 10 samples: {path}"
            )


        nonempty = sum(
            bool(
                str(
                    sample.get(
                        "text",
                        "",
                    )
                ).strip()
            )
            for sample
            in samples
        )


        if nonempty < 8:

            raise RuntimeError(
                f"Too few usable samples: {path}"
            )


        total += 1


print(
    f"Validated {total} generation files."
)

print(
    "Deep cache check: PASS"
)

PY

echo


# ============================================================
# RUN SELFCHECKGPT
# ============================================================

echo "============================================================"
echo "STARTING SELFCHECKGPT-NLI"
echo "============================================================"
echo


set +e

python -u "$SCRIPT" \
    --models "$MODEL" \
    --generation-root "$GENERATION_ROOT" \
    --bridge-root "$BRIDGE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --log-dir "$LOG_DIR"

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
    echo "SELFCHECKGPT-NLI FAILED"
    echo "============================================================"
    echo
    echo "Model:"
    echo "$MODEL"
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
# VERIFY OUTPUT
# ============================================================

MODEL_OUTPUT="$OUTPUT_ROOT/$MODEL"


if [ ! -f "$MODEL_OUTPUT/summary.csv" ]; then

    echo "ERROR:"
    echo "Missing:"
    echo "$MODEL_OUTPUT/summary.csv"

    exit 1

fi


if [ ! -f "$MODEL_OUTPUT/healthbench_test_scores.csv" ]; then

    echo "ERROR:"
    echo "Missing:"
    echo "$MODEL_OUTPUT/healthbench_test_scores.csv"

    exit 1

fi


if [ ! -f "$MODEL_OUTPUT/clindet_external_test_scores.csv" ]; then

    echo "ERROR:"
    echo "Missing:"
    echo "$MODEL_OUTPUT/clindet_external_test_scores.csv"

    exit 1

fi


if [ "$VAL_COUNT" -eq 123 ]; then

    if [ ! -f "$MODEL_OUTPUT/healthbench_validation_scores.csv" ]; then

        echo "ERROR:"
        echo "Missing validation scores."

        exit 1

    fi


    if [ ! -f "$MODEL_OUTPUT/threshold_validation.csv" ]; then

        echo "ERROR:"
        echo "Missing validation threshold file."

        exit 1

    fi

fi


# ============================================================
# PRINT RESULTS
# ============================================================

echo "============================================================"
echo "SELFCHECKGPT RESULTS"
echo "============================================================"

python - <<PY

import pandas as pd

path = (
    "$MODEL_OUTPUT/summary.csv"
)

df = pd.read_csv(
    path
)

print(
    df.to_string(
        index=False
    )
)

PY


# ============================================================
# COMPLETE
# ============================================================

echo
echo "============================================================"
echo "SELFCHECKGPT COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo "Model:    $MODEL"
echo
echo "Results:"
echo "$MODEL_OUTPUT/summary.csv"
echo
echo "HealthBench scores:"
echo "$MODEL_OUTPUT/healthbench_test_scores.csv"
echo
echo "ClinDet scores:"
echo "$MODEL_OUTPUT/clindet_external_test_scores.csv"

if [ "$VAL_COUNT" -eq 123 ]; then

    echo
    echo "Validation scores:"
    echo "$MODEL_OUTPUT/healthbench_validation_scores.csv"

    echo
    echo "Threshold search:"
    echo "$MODEL_OUTPUT/threshold_validation.csv"

fi

echo
echo "============================================================"