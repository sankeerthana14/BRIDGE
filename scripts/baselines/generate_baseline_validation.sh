#!/bin/bash

#SBATCH --job-name=baseline_val
#SBATCH --gres=gpu:1
#SBATCH --partition=RTXA6Kq
#SBATCH --nodelist=node10
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

SCRIPT="$REPO_ROOT/scripts/generate_baseline_validation.py"

OUTPUT_ROOT="$REPO_ROOT/cache/baseline_generations"

HEALTHBENCH_MANIFEST="$REPO_ROOT/data/manifests/healthbench_evidence_sufficiency_v1.csv"

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
    echo "sbatch slurm/generate_baseline_validation.sh MODEL"
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
        echo "Unknown model:"
        echo "$MODEL"

        exit 1
        ;;

esac


# ============================================================
# GENERATION CONFIGURATION
# ============================================================

NUM_SAMPLES=10

TEMPERATURE=1.0

TOP_P=0.95

MAX_NEW_TOKENS=256

SEED=42


# ============================================================
# RUN INFORMATION
# ============================================================

echo "============================================================"
echo "BRIDGE BASELINE VALIDATION GENERATION"
echo "============================================================"
echo "Start time:          $(date)"
echo "Host:                $(hostname)"
echo "SLURM job ID:        ${SLURM_JOB_ID:-N/A}"
echo "Model:               $MODEL"
echo
echo "Dataset:             HealthBench validation"
echo "Expected examples:   123"
echo "Expected class 0:    78"
echo "Expected class 1:    45"
echo
echo "Repository root:     $REPO_ROOT"
echo "Script:              $SCRIPT"
echo "Output root:         $OUTPUT_ROOT"
echo "Manifest:            $HEALTHBENCH_MANIFEST"
echo
echo "Samples/example:     $NUM_SAMPLES"
echo "Temperature:         $TEMPERATURE"
echo "Top-p:               $TOP_P"
echo "Max new tokens:      $MAX_NEW_TOKENS"
echo "Seed:                $SEED"
echo "============================================================"
echo


# ============================================================
# GPU INFORMATION
# ============================================================

echo "============================================================"
echo "GPU INFORMATION"
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
import torch
import transformers
import numpy
import pandas

print(
    "Python:",
    sys.version
)

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


# Verify Lingshu architecture class exists
from transformers import (
    Qwen2_5_VLForConditionalGeneration
)

print(
    "Qwen2.5-VL model class import: PASS"
)

print()
print(
    "Environment check: PASS"
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
    echo "Script does not exist:"
    echo "$SCRIPT"

    exit 1

fi


python -m py_compile "$SCRIPT"

echo "Syntax check: PASS"

echo


# ============================================================
# LINGSHU ARCHITECTURE CHECK
# ============================================================

if [ "$MODEL" = "lingshu" ]; then

    echo "============================================================"
    echo "VERIFYING LINGSHU QWEN2.5-VL CODE PATH"
    echo "============================================================"


    if ! grep -q \
        'Qwen2_5_VLForConditionalGeneration' \
        "$SCRIPT"; then

        echo "ERROR:"
        echo "Lingshu Qwen2.5-VL model class not found"
        echo "in validation-generation script."

        exit 1

    fi


    if ! grep -q \
        'format_lingshu_prompt' \
        "$SCRIPT"; then

        echo "ERROR:"
        echo "Lingshu prompt formatter not found."

        exit 1

    fi


    if ! grep -q \
        'model_name == "lingshu"' \
        "$SCRIPT"; then

        echo "ERROR:"
        echo "Lingshu-specific code path not found."

        exit 1

    fi


    echo "Lingshu architecture handling: PASS"
    echo

fi


# ============================================================
# MANIFEST CHECK
# ============================================================

echo "============================================================"
echo "CHECKING HEALTHBENCH MANIFEST"
echo "============================================================"


if [ ! -f "$HEALTHBENCH_MANIFEST" ]; then

    echo "ERROR:"
    echo "HealthBench manifest missing:"
    echo "$HEALTHBENCH_MANIFEST"

    exit 1

fi


python - <<PY

import pandas as pd

path = "$HEALTHBENCH_MANIFEST"

df = pd.read_csv(
    path
)

included = (
    df["included"]
    .astype(str)
    .str.lower()
    .isin(
        [
            "true",
            "1",
            "yes",
        ]
    )
)

validation = (
    df["split"]
    .astype(str)
    .str.lower()
    == "validation"
)

val = df[
    included
    &
    validation
].copy()


print(
    "Manifest total rows:",
    len(df)
)

print(
    "Validation rows:",
    len(val)
)


counts = (
    val[
        "binary_label"
    ]
    .astype(int)
    .value_counts()
    .sort_index()
    .to_dict()
)


print(
    "Validation class counts:",
    counts
)


if len(val) != 123:

    raise RuntimeError(
        f"Expected 123 validation rows, "
        f"found {len(val)}."
    )


if counts.get(
    0,
    0
) != 78:

    raise RuntimeError(
        "Expected 78 sufficient "
        "validation examples."
    )


if counts.get(
    1,
    0
) != 45:

    raise RuntimeError(
        "Expected 45 insufficient "
        "validation examples."
    )


print()
print(
    "Manifest check: PASS"
)

PY

echo


# ============================================================
# EXISTING CACHE CHECK
# ============================================================

VAL_ROOT="$OUTPUT_ROOT/$MODEL/healthbench/validation"


if [ -d "$VAL_ROOT" ]; then

    EXISTING=$(find "$VAL_ROOT" \
        -type f \
        -name "*.pt" \
        | wc -l)

else

    EXISTING=0

fi


echo "============================================================"
echo "EXISTING VALIDATION CACHE"
echo "============================================================"

echo "Model: $MODEL"
echo "Files: $EXISTING / 123"

echo


if [ "$EXISTING" -eq 123 ]; then

    echo "Validation cache already contains 123 files."
    echo
    echo "The Python script will still validate the cache"
    echo "if run manually, but no generation is necessary."

    exit 0

fi


if [ "$EXISTING" -gt 123 ]; then

    echo "ERROR:"
    echo "Validation cache contains more than 123 files."

    exit 1

fi


echo "Existing valid files will be skipped."
echo "Invalid or missing files will be generated."
echo


# ============================================================
# START GENERATION
# ============================================================

echo "============================================================"
echo "STARTING VALIDATION GENERATION"
echo "============================================================"
echo


set +e


python -u "$SCRIPT" \
    --models "$MODEL" \
    --healthbench-manifest "$HEALTHBENCH_MANIFEST" \
    --output-root "$OUTPUT_ROOT" \
    --num-samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --seed "$SEED"


EXIT_CODE=$?


set -e


echo
echo "============================================================"
echo "GENERATION PROCESS FINISHED"
echo "============================================================"

echo "Exit code: $EXIT_CODE"

echo


if [ "$EXIT_CODE" -ne 0 ]; then

    echo "============================================================"
    echo "VALIDATION GENERATION FAILED"
    echo "============================================================"
    echo
    echo "Model:"
    echo "$MODEL"
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
# FINAL FILE COUNT
# ============================================================

FINAL_COUNT=$(find "$VAL_ROOT" \
    -type f \
    -name "*.pt" \
    | wc -l)


echo "============================================================"
echo "FINAL CACHE CHECK"
echo "============================================================"

echo "Validation files: $FINAL_COUNT / 123"

echo


if [ "$FINAL_COUNT" -ne 123 ]; then

    echo "ERROR:"
    echo "Expected 123 validation files."
    echo "Found $FINAL_COUNT."

    exit 1

fi


# ============================================================
# DEEP VALIDATION
# ============================================================

echo "============================================================"
echo "DEEP VALIDATION"
echo "============================================================"


python - <<PY

from pathlib import Path

import numpy as np
import torch


root = Path(
    "$VAL_ROOT"
)

files = sorted(
    root.glob(
        "*.pt"
    )
)


if len(files) != 123:

    raise RuntimeError(
        f"Expected 123 files, "
        f"found {len(files)}."
    )


required = {
    "example_id",
    "label",
    "prompt",
    "greedy",
    "samples",
    "generation_config",
}


missing = 0
bad_samples = 0
empty_greedy = 0
too_few_samples = 0

labels = []


for path in files:

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


    if not required.issubset(
        obj.keys()
    ):

        missing += 1
        continue


    labels.append(
        int(
            obj[
                "label"
            ]
        )
    )


    if len(
        obj[
            "samples"
        ]
    ) != 10:

        bad_samples += 1


    greedy = str(
        obj[
            "greedy"
        ].get(
            "text",
            "",
        )
    ).strip()


    if not greedy:

        empty_greedy += 1


    nonempty = sum(
        bool(
            str(
                item.get(
                    "text",
                    "",
                )
            ).strip()
        )

        for item
        in obj[
            "samples"
        ]
    )


    if nonempty < 8:

        too_few_samples += 1


labels = np.asarray(
    labels
)


counts = {
    0: int(
        np.sum(
            labels == 0
        )
    ),

    1: int(
        np.sum(
            labels == 1
        )
    ),
}


print(
    "Files:",
    len(files)
)

print(
    "Missing required keys:",
    missing
)

print(
    "Bad sample count:",
    bad_samples
)

print(
    "Empty greedy responses:",
    empty_greedy
)

print(
    "Examples with <8 usable samples:",
    too_few_samples
)

print(
    "Class counts:",
    counts
)


if missing != 0:

    raise RuntimeError(
        "Missing required fields."
    )


if bad_samples != 0:

    raise RuntimeError(
        "Incorrect sample counts."
    )


if empty_greedy != 0:

    raise RuntimeError(
        "Empty greedy generations."
    )


if too_few_samples != 0:

    raise RuntimeError(
        "Too few usable stochastic samples."
    )


if counts != {
    0: 78,
    1: 45,
}:

    raise RuntimeError(
        f"Unexpected class counts: {counts}"
    )


print()
print(
    "Deep validation: PASS"
)

PY


# ============================================================
# SAMPLE OUTPUT
# ============================================================

echo
echo "============================================================"
echo "SAMPLE VALIDATION OUTPUT"
echo "============================================================"


python - <<PY

from pathlib import Path
import torch


root = Path(
    "$VAL_ROOT"
)

path = sorted(
    root.glob(
        "*.pt"
    )
)[0]


obj = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)


print(
    "File:",
    path
)

print()

print(
    "Example ID:",
    obj[
        "example_id"
    ]
)

print()

print(
    "Label:",
    obj[
        "label"
    ]
)

print()

print(
    "Greedy response:"
)

print(
    repr(
        obj[
            "greedy"
        ][
            "text"
        ][:600]
    )
)

print()

print(
    "First stochastic response:"
)

print(
    repr(
        obj[
            "samples"
        ][0][
            "text"
        ][:600]
    )
)

PY


# ============================================================
# COMPLETE
# ============================================================

echo
echo "============================================================"
echo "VALIDATION GENERATION COMPLETE"
echo "============================================================"

echo "End time: $(date)"
echo "Model:    $MODEL"

echo

echo "Cache:"
echo "$VAL_ROOT"

echo

echo "Next step:"
echo "sbatch slurm/selfcheckgpt.sh $MODEL"

echo
echo "============================================================"