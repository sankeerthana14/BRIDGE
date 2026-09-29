#!/bin/bash

#SBATCH --job-name=baseline_generate
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

SCRIPT="$REPO_ROOT/scripts/generate_baseline_samples.py"

ACTIVATION_ROOT="$REPO_ROOT/cache/activations"

OUTPUT_ROOT="$REPO_ROOT/cache/baseline_generations"

HEALTHBENCH_MANIFEST="$REPO_ROOT/data/manifests/healthbench_evidence_sufficiency_v1.csv"

CLINDET_MANIFEST="$REPO_ROOT/data/manifests/clindet_evidence_sufficiency_v1.csv"

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
    echo "sbatch slurm/generate_baselines.sh MODEL"
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
# GENERATION CONFIG
# ============================================================

NUM_SAMPLES=10

TEMPERATURE=1.0

TOP_P=0.95

MAX_NEW_TOKENS=256

SEED=42


# ============================================================
# INFO
# ============================================================

echo "============================================================"
echo "BRIDGE BASELINE GENERATION CACHE"
echo "============================================================"
echo "Start time:          $(date)"
echo "Host:                $(hostname)"
echo "SLURM job ID:        ${SLURM_JOB_ID:-N/A}"
echo "Model:               $MODEL"
echo
echo "Repository root:     $REPO_ROOT"
echo "Script:              $SCRIPT"
echo "Activation root:     $ACTIVATION_ROOT"
echo "Output root:         $OUTPUT_ROOT"
echo
echo "HealthBench manifest:"
echo "$HEALTHBENCH_MANIFEST"
echo
echo "ClinDet manifest:"
echo "$CLINDET_MANIFEST"
echo
echo "Samples/example:     $NUM_SAMPLES"
echo "Temperature:         $TEMPERATURE"
echo "Top-p:               $TOP_P"
echo "Max new tokens:      $MAX_NEW_TOKENS"
echo "Seed:                $SEED"
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

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA unavailable."
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0)
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
    echo "$SCRIPT does not exist."

    exit 1

fi


python -m py_compile "$SCRIPT"

echo "Syntax check: PASS"

echo


# ============================================================
# IMPORTANT OPENBIOLLM CODE CHECK
# ============================================================

if [ "$MODEL" = "openbiollm" ]; then

    echo "============================================================"
    echo "VERIFYING OPENBIOLLM LLAMA-3 CODE PATH"
    echo "============================================================"

    if ! grep -q 'if model_name == "openbiollm"' "$SCRIPT"; then

        echo "ERROR:"
        echo "OpenBioLLM-specific code path not found."
        echo "You are probably running an old Python script."

        exit 1

    fi

    if ! grep -q '<|begin_of_text|>' "$SCRIPT"; then

        echo "ERROR:"
        echo "Explicit Llama-3 prompt format not found."

        exit 1

    fi

    if ! grep -q '<|start_header_id|>' "$SCRIPT"; then

        echo "ERROR:"
        echo "Llama-3 header tokens not found."

        exit 1

    fi

    if ! grep -q '<|eot_id|>' "$SCRIPT"; then

        echo "ERROR:"
        echo "Llama-3 EOT token handling not found."

        exit 1

    fi

    echo "OpenBioLLM code path: PASS"
    echo

fi


# ============================================================
# MANIFEST CHECK
# ============================================================

echo "============================================================"
echo "CHECKING MANIFESTS"
echo "============================================================"

if [ ! -f "$HEALTHBENCH_MANIFEST" ]; then

    echo "ERROR:"
    echo "HealthBench manifest missing."

    exit 1

fi


if [ ! -f "$CLINDET_MANIFEST" ]; then

    echo "ERROR:"
    echo "ClinDet manifest missing."

    exit 1

fi


python - <<PY

import pandas as pd

hb = pd.read_csv(
    "$HEALTHBENCH_MANIFEST"
)

cd = pd.read_csv(
    "$CLINDET_MANIFEST"
)

print(
    "HealthBench manifest rows:",
    len(hb)
)

print(
    "ClinDet manifest rows:",
    len(cd)
)

if len(hb) != 6000:

    raise RuntimeError(
        f"Expected 6000 HealthBench rows, "
        f"found {len(hb)}"
    )

if len(cd) != 94:

    raise RuntimeError(
        f"Expected 94 ClinDet rows, "
        f"found {len(cd)}"
    )

print()
print(
    "Manifest check: PASS"
)

PY

echo


# ============================================================
# ACTIVATION CHECK
# ============================================================

echo "============================================================"
echo "CHECKING ACTIVATION CACHE"
echo "============================================================"

MODEL_ACTIVATION_ROOT="$ACTIVATION_ROOT/$MODEL"


if [ ! -d "$MODEL_ACTIVATION_ROOT" ]; then

    echo "ERROR:"
    echo "Activation root missing:"
    echo "$MODEL_ACTIVATION_ROOT"

    exit 1

fi


NUM_ACTIVATIONS=$(find "$MODEL_ACTIVATION_ROOT" \
    -type f \
    -name "*.pt" \
    | wc -l)


echo "Activation files: $NUM_ACTIVATIONS / 897"


if [ "$NUM_ACTIVATIONS" -ne 897 ]; then

    echo "ERROR:"
    echo "Expected 897 activation files."

    exit 1

fi


echo "Activation cache: PASS"

echo


# ============================================================
# EXISTING CACHE
# ============================================================

echo "============================================================"
echo "EXISTING GENERATION CACHE STATUS"
echo "============================================================"


HB_OUTPUT="$OUTPUT_ROOT/$MODEL/healthbench/test"

CLINDET_OUTPUT="$OUTPUT_ROOT/$MODEL/clindet/external_test"


if [ -d "$HB_OUTPUT" ]; then

    HB_EXISTING=$(find "$HB_OUTPUT" \
        -type f \
        -name "*.pt" \
        | wc -l)

else

    HB_EXISTING=0

fi


if [ -d "$CLINDET_OUTPUT" ]; then

    CLINDET_EXISTING=$(find "$CLINDET_OUTPUT" \
        -type f \
        -name "*.pt" \
        | wc -l)

else

    CLINDET_EXISTING=0

fi


TOTAL_EXISTING=$((HB_EXISTING + CLINDET_EXISTING))


echo "HealthBench existing: $HB_EXISTING / 188"
echo "ClinDet existing:     $CLINDET_EXISTING / 94"
echo "Total existing:       $TOTAL_EXISTING / 282"
echo
echo "Valid cache files will be skipped."
echo "Invalid cache files will be regenerated."
echo


# ============================================================
# RUN
# ============================================================

echo "============================================================"
echo "STARTING BASELINE GENERATION"
echo "============================================================"
echo


set +e

python -u "$SCRIPT" \
    --models "$MODEL" \
    --activation-root "$ACTIVATION_ROOT" \
    --healthbench-manifest "$HEALTHBENCH_MANIFEST" \
    --clindet-source "$CLINDET_MANIFEST" \
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
    echo "BASELINE GENERATION FAILED"
    echo "============================================================"

    exit "$EXIT_CODE"

fi


# ============================================================
# FILE COUNT VALIDATION
# ============================================================

echo "============================================================"
echo "VERIFYING OUTPUT FILE COUNTS"
echo "============================================================"


HB_COUNT=$(find "$HB_OUTPUT" \
    -type f \
    -name "*.pt" \
    | wc -l)


CLINDET_COUNT=$(find "$CLINDET_OUTPUT" \
    -type f \
    -name "*.pt" \
    | wc -l)


TOTAL_COUNT=$((HB_COUNT + CLINDET_COUNT))


echo "HealthBench: $HB_COUNT / 188"
echo "ClinDet:     $CLINDET_COUNT / 94"
echo "Total:       $TOTAL_COUNT / 282"


if [ "$HB_COUNT" -ne 188 ]; then

    echo "ERROR:"
    echo "HealthBench cache incomplete."

    exit 1

fi


if [ "$CLINDET_COUNT" -ne 94 ]; then

    echo "ERROR:"
    echo "ClinDet cache incomplete."

    exit 1

fi


if [ "$TOTAL_COUNT" -ne 282 ]; then

    echo "ERROR:"
    echo "Expected 282 total files."

    exit 1

fi


echo
echo "File count validation: PASS"
echo


# ============================================================
# DEEP VALIDATION
# ============================================================

echo "============================================================"
echo "DEEP GENERATION VALIDATION"
echo "============================================================"


python - <<PY

from pathlib import Path
import torch

root = (
    Path("$OUTPUT_ROOT")
    / "$MODEL"
)

files = sorted(
    root.rglob("*.pt")
)

if len(files) != 282:

    raise RuntimeError(
        f"Expected 282 files, "
        f"found {len(files)}"
    )


missing_required = 0
bad_sample_count = 0
empty_greedy = 0
all_samples_empty = 0
any_sample_empty = 0
too_few_nonempty = 0


required = {
    "example_id",
    "label",
    "prompt",
    "greedy",
    "samples",
    "generation_config",
}


for path in files:

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not required.issubset(
        obj.keys()
    ):

        missing_required += 1
        continue


    samples = obj["samples"]


    if len(samples) != 10:

        bad_sample_count += 1
        continue


    greedy_text = str(
        obj[
            "greedy"
        ].get(
            "text",
            "",
        )
    ).strip()


    if not greedy_text:

        empty_greedy += 1


    sample_texts = [
        str(
            sample.get(
                "text",
                "",
            )
        ).strip()

        for sample
        in samples
    ]


    nonempty = sum(
        bool(text)
        for text
        in sample_texts
    )


    if nonempty == 0:

        all_samples_empty += 1


    if nonempty < 10:

        any_sample_empty += 1


    if nonempty < 8:

        too_few_nonempty += 1


print(
    "Files:",
    len(files)
)

print(
    "Missing required keys:",
    missing_required
)

print(
    "Bad sample count:",
    bad_sample_count
)

print(
    "Empty greedy responses:",
    empty_greedy
)

print(
    "All stochastic responses empty:",
    all_samples_empty
)

print(
    "Examples with any empty stochastic sample:",
    any_sample_empty
)

print(
    "Examples with <8 non-empty samples:",
    too_few_nonempty
)


if missing_required != 0:

    raise RuntimeError(
        "Missing required cache fields."
    )


if bad_sample_count != 0:

    raise RuntimeError(
        "Incorrect stochastic sample counts."
    )


if empty_greedy != 0:

    raise RuntimeError(
        f"{empty_greedy} examples "
        "have empty greedy responses."
    )


if all_samples_empty != 0:

    raise RuntimeError(
        f"{all_samples_empty} examples "
        "have no stochastic responses."
    )


if too_few_nonempty != 0:

    raise RuntimeError(
        f"{too_few_nonempty} examples "
        "have fewer than 8 usable samples."
    )


print()
print(
    "Deep generation validation: PASS"
)

PY

echo


# ============================================================
# SAMPLE OUTPUT
# ============================================================

echo "============================================================"
echo "SAMPLE OUTPUT"
echo "============================================================"


python - <<PY

from pathlib import Path
import torch

root = (
    Path("$OUTPUT_ROOT")
    / "$MODEL"
)

files = sorted(
    root.rglob("*.pt")
)

obj = torch.load(
    files[0],
    map_location="cpu",
    weights_only=False,
)

print(
    "File:",
    files[0]
)

print()

print(
    "Example ID:",
    obj["example_id"]
)

print()

print(
    "Greedy text:"
)

print(
    repr(
        obj[
            "greedy"
        ]["text"][:600]
    )
)

print()

print(
    "Greedy IDs:"
)

print(
    obj[
        "greedy"
    ][
        "generated_token_ids"
    ][:30]
)

print()

print(
    "First stochastic text:"
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

print()

print(
    "First stochastic IDs:"
)

print(
    obj[
        "samples"
    ][0][
        "generated_token_ids"
    ][:30]
)

PY

echo


# ============================================================
# COMPLETE
# ============================================================

echo
echo "============================================================"
echo "BASELINE GENERATION COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo "Model:    $MODEL"
echo
echo "Cache:"
echo "$OUTPUT_ROOT/$MODEL"
echo
echo "============================================================"