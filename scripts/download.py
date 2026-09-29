"""
BRIDGE Revision — Download Script
=================================

Downloads the models, datasets, and baseline repositories needed
for the JBHI major revision.

PRIMARY REVISION SET
--------------------

Target / evaluation models:
    - BioMistral-7B
    - Llama3-OpenBioLLM-8B
    - Llama-3-8B-UltraMedical
    - MedGemma 1.5 4B
    - Lingshu-7B

Primary datasets:
    - HealthBench
    - ClinDet-Bench

Primary baseline repositories:
    - INSIDE / EigenScore
    - SelfCheckGPT
    - Semantic Entropy / Semantic Entropy Probes
    - ICR Probe
    - SAPLMA-style baseline (implemented locally)

Optional datasets:
    - MediQ
    - PubMedQA

Usage
-----

# Show status
python download.py --status

# Download all PRIMARY revision resources
python download.py --all

# Download only models
python download.py --models

# Download one model
python download.py --models --name medgemma

# Download datasets
python download.py --datasets

# Download one dataset
python download.py --datasets --name clindet

# Clone baseline repos
python download.py --baselines

# Clone one baseline
python download.py --baselines --name icr_probe

# Include optional datasets (MediQ + PubMedQA)
python download.py --datasets --include-optional

# Everything including optional resources
python download.py --all --include-optional

# Dry run
python download.py --all --dry-run

# Verify
python download.py --verify


Prerequisites
-------------

pip install huggingface_hub datasets

Authenticate:
    huggingface-cli login

MedGemma is gated. Accept its terms first:
    https://huggingface.co/google/medgemma-1.5-4b-it
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PATHS import (
    MODELS_DIR,
    DATA_DIR,
    BASELINES_DIR,
)


# ============================================================
# RESOURCE DEFINITIONS
# ============================================================

# ------------------------------------------------------------
# TARGET / EVALUATION MODELS
#
# These are the LLMs whose hidden states BRIDGE will probe.
# They are NOT "baselines".
# ------------------------------------------------------------

MODEL_SPECS = {
    # Original manuscript models
    "biomistral": {
        "hf_id": "BioMistral/BioMistral-7B",
        "dir_name": "biomistral",
        "gated": False,
        "role": "original",
    },

    "openbiollm": {
        "hf_id": "aaditya/Llama3-OpenBioLLM-8B",
        "dir_name": "openbiollm",
        "gated": False,
        "role": "original",
    },

    "ultramedical": {
        "hf_id": "TsinghuaC3I/Llama-3-8B-UltraMedical",
        "dir_name": "ultramedical",
        "gated": False,
        "role": "original",
    },

    # Reviewer-requested / newer models
    "medgemma": {
        "hf_id": "google/medgemma-1.5-4b-it",
        "dir_name": "medgemma",
        "gated": True,
        "role": "new",
    },

    "lingshu": {
        "hf_id": "lingshu-medical-mllm/Lingshu-7B",
        "dir_name": "lingshu",
        "gated": False,
        "role": "new",
    },
}


# ------------------------------------------------------------
# DATASETS
#
# source_type:
#
#   hf_snapshot
#       Download the actual raw HuggingFace dataset repository.
#
#   hf_dataset
#       Use datasets.load_dataset() and save_to_disk().
#
#   git
#       Clone the benchmark repository.
#
# "optional": True means it is NOT downloaded by --all unless
# --include-optional is provided.
# ------------------------------------------------------------

DATASET_SPECS = {
    "healthbench": {
        "source_type": "hf_snapshot",
        "hf_id": "openai/healthbench",
        "dir_name": "healthbench",
        "optional": False,
    },

    # PRIMARY external dataset for the revision
    "clindet": {
        "source_type": "git",
        "git_url": (
            "https://github.com/"
            "yusukewatanabe1208/ClinDet_Benchmark.git"
        ),
        "dir_name": "clindet",
        "optional": False,
    },

    # Backup / optional external benchmark
    "mediq": {
        "source_type": "git",
        "git_url": "https://github.com/stellalisy/mediQ.git",
        "dir_name": "mediq",
        "optional": True,
    },

    # Keep available, but NOT primary Dataset 2
    "pubmedqa": {
        "source_type": "hf_dataset",
        "hf_id": "qiaojin/PubMedQA",
        "subset": "pqa_labeled",
        "dir_name": "pubmedqa",
        "optional": True,
    },
}


# ------------------------------------------------------------
# BASELINES
#
# These are competing DETECTION methods.
# They are not target LLMs.
# ------------------------------------------------------------

BASELINE_SPECS = {
    "eigenscore": {
        "git_url": "https://github.com/D2I-ai/eigenscore.git",
        "dir_name": "eigenscore",
    },

    "selfcheckgpt": {
        "git_url": "https://github.com/potsawee/selfcheckgpt.git",
        "dir_name": "selfcheckgpt",
    },

    "semantic_entropy": {
        "git_url": (
            "https://github.com/"
            "OATML/semantic-entropy-probes.git"
        ),
        "dir_name": "semantic-entropy-probes",
    },

    # Strong recent hidden-state detector for related work /
    # optional experimental comparison if time permits.
    "icr_probe": {
        "git_url": (
            "https://github.com/"
            "XavierZhang2002/ICR_Probe.git"
        ),
        "dir_name": "icr-probe",
    },
}


# ============================================================
# PATH HELPERS
# ============================================================

def model_dir(name):
    return os.path.join(
        MODELS_DIR,
        MODEL_SPECS[name]["dir_name"],
    )


def dataset_dir(name):
    return os.path.join(
        DATA_DIR,
        DATASET_SPECS[name]["dir_name"],
    )


def baseline_dir(name):
    return os.path.join(
        BASELINES_DIR,
        BASELINE_SPECS[name]["dir_name"],
    )


SAPLMA_DIR = os.path.join(
    BASELINES_DIR,
    "saplma",
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(
        f"[{timestamp}] [{level}] {msg}"
    )


def dir_has_content(path):
    """Return True if a directory exists and contains files."""
    return (
        os.path.isdir(path)
        and len(os.listdir(path)) > 0
    )


def ensure_dir(path):
    """Create a directory if necessary."""
    os.makedirs(
        path,
        exist_ok=True,
    )


def run_git_clone(
    git_url,
    local_dir,
    dry_run=False,
):
    """Clone a Git repository."""

    if dir_has_content(local_dir):
        log(
            f"SKIP — already exists at {local_dir}"
        )
        return True

    log(f"  Repository: {git_url}")
    log(f"  Target:     {local_dir}")

    if dry_run:
        log(
            "  [DRY RUN] Would clone repository"
        )
        return True

    try:
        ensure_dir(
            os.path.dirname(local_dir)
        )

        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                git_url,
                local_dir,
            ],
            check=True,
        )

        return True

    except subprocess.CalledProcessError as e:
        log(
            f"Git clone failed: {e}",
            "ERROR",
        )
        return False


# ============================================================
# HUGGINGFACE LOGIN
# ============================================================

def check_hf_login():
    """
    Verify HuggingFace authentication.

    Non-gated repositories may still download without login.
    """

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        user = api.whoami()

        username = (
            user.get("name")
            or user.get("fullname")
            or "unknown"
        )

        log(
            f"HuggingFace login OK — {username}"
        )

        return True

    except Exception as e:

        log(
            f"HuggingFace login unavailable: {e}",
            "WARN",
        )

        log(
            "Run: huggingface-cli login",
            "WARN",
        )

        return False


# ============================================================
# MODEL DOWNLOADS
# ============================================================

def download_model(
    name,
    dry_run=False,
):
    """Download one target/evaluation model."""

    spec = MODEL_SPECS[name]

    hf_id = spec["hf_id"]
    local_dir = model_dir(name)

    if dir_has_content(local_dir):

        log(
            f"SKIP {name} — already exists at "
            f"{local_dir}"
        )

        return True

    log("")
    log(f"Downloading model: {name}")
    log(f"  Role:            {spec['role']}")
    log(f"  HuggingFace ID:  {hf_id}")
    log(f"  Target:          {local_dir}")

    if spec["gated"]:

        log(
            "  ⚠ GATED MODEL — make sure access "
            "has been accepted on HuggingFace"
        )

        log(
            f"  https://huggingface.co/{hf_id}"
        )

    if dry_run:

        log(
            f"  [DRY RUN] Would download {name}"
        )

        return True

    try:
        from huggingface_hub import snapshot_download

        ensure_dir(local_dir)

        snapshot_download(
            repo_id=hf_id,
            repo_type="model",
            local_dir=local_dir,
        )

        log(
            f"  ✓ {name} downloaded"
        )

        return True

    except Exception as e:

        log(
            f"  ✗ Failed to download {name}: {e}",
            "ERROR",
        )

        if spec["gated"]:

            log(
                f"  Check access at "
                f"https://huggingface.co/{hf_id}",
                "ERROR",
            )

        return False


def download_all_models(
    name_filter=None,
    dry_run=False,
):
    """Download all revision target models."""

    log("=" * 60)
    log("DOWNLOADING TARGET / EVALUATION MODELS")
    log("=" * 60)

    ensure_dir(MODELS_DIR)

    results = {}

    for name in MODEL_SPECS:

        if (
            name_filter
            and name != name_filter
        ):
            continue

        ok = download_model(
            name,
            dry_run=dry_run,
        )

        results[name] = (
            "success"
            if ok
            else "failed"
        )

    if (
        name_filter
        and name_filter not in MODEL_SPECS
    ):

        log(
            f"Unknown model name: {name_filter}",
            "WARN",
        )

    return results


# ============================================================
# DATASET DOWNLOADS
# ============================================================

def download_hf_dataset_snapshot(
    name,
    spec,
    local_dir,
    dry_run=False,
):
    """
    Download raw files from a HuggingFace dataset repository.

    We use this for HealthBench because we need the raw
    OSS-Eval + Hard JSONL files, not only one default config.
    """

    hf_id = spec["hf_id"]

    log(
        f"  HuggingFace dataset repo: {hf_id}"
    )

    if dry_run:

        log(
            f"  [DRY RUN] Would snapshot {hf_id}"
        )

        return True

    try:
        from huggingface_hub import snapshot_download

        ensure_dir(local_dir)

        snapshot_download(
            repo_id=hf_id,
            repo_type="dataset",
            local_dir=local_dir,
        )

        return True

    except Exception as e:

        log(
            f"  ✗ HuggingFace snapshot failed: {e}",
            "ERROR",
        )

        return False


def download_hf_dataset_processed(
    name,
    spec,
    local_dir,
    dry_run=False,
):
    """
    Download using HuggingFace datasets.load_dataset()
    and persist via save_to_disk().
    """

    hf_id = spec["hf_id"]
    subset = spec.get("subset")

    log(f"  HuggingFace ID: {hf_id}")

    if subset:
        log(f"  Subset:         {subset}")

    if dry_run:

        log(
            f"  [DRY RUN] Would download {hf_id}"
        )

        return True

    try:
        from datasets import load_dataset

        if subset:

            ds = load_dataset(
                hf_id,
                subset,
            )

        else:

            ds = load_dataset(
                hf_id,
            )

        ensure_dir(local_dir)

        ds.save_to_disk(
            local_dir
        )

        for split_name, split_data in ds.items():

            log(
                f"  Split '{split_name}': "
                f"{len(split_data)} examples"
            )

            log(
                f"    Columns: "
                f"{split_data.column_names}"
            )

        return True

    except Exception as e:

        log(
            f"  ✗ Dataset download failed: {e}",
            "ERROR",
        )

        return False


def download_dataset(
    name,
    dry_run=False,
):
    """Download one configured dataset."""

    spec = DATASET_SPECS[name]
    local_dir = dataset_dir(name)

    if dir_has_content(local_dir):

        log(
            f"SKIP {name} — already exists at "
            f"{local_dir}"
        )

        return True

    log("")
    log(f"Downloading dataset: {name}")
    log(f"  Target: {local_dir}")

    source_type = spec["source_type"]

    if source_type == "hf_snapshot":

        ok = download_hf_dataset_snapshot(
            name,
            spec,
            local_dir,
            dry_run=dry_run,
        )

    elif source_type == "hf_dataset":

        ok = download_hf_dataset_processed(
            name,
            spec,
            local_dir,
            dry_run=dry_run,
        )

    elif source_type == "git":

        ok = run_git_clone(
            spec["git_url"],
            local_dir,
            dry_run=dry_run,
        )

    else:

        log(
            f"Unknown dataset source type: "
            f"{source_type}",
            "ERROR",
        )

        return False

    if ok:

        log(
            f"  ✓ {name} downloaded"
        )

    return ok


def download_all_datasets(
    name_filter=None,
    dry_run=False,
    include_optional=False,
):
    """Download the dataset suite."""

    log("=" * 60)
    log("DOWNLOADING DATASETS")
    log("=" * 60)

    ensure_dir(DATA_DIR)

    results = {}

    for name, spec in DATASET_SPECS.items():

        if (
            name_filter
            and name != name_filter
        ):
            continue

        if (
            spec.get("optional", False)
            and not include_optional
            and name_filter is None
        ):

            log(
                f"SKIP optional dataset: {name}"
            )

            continue

        ok = download_dataset(
            name,
            dry_run=dry_run,
        )

        results[name] = (
            "success"
            if ok
            else "failed"
        )

    if (
        name_filter
        and name_filter not in DATASET_SPECS
    ):

        log(
            f"Unknown dataset name: "
            f"{name_filter}",
            "WARN",
        )

    return results


# ============================================================
# BASELINE REPOSITORIES
# ============================================================

def clone_baseline(
    name,
    dry_run=False,
):
    """Clone one baseline repository."""

    spec = BASELINE_SPECS[name]

    git_url = spec["git_url"]
    local_dir = baseline_dir(name)

    if dir_has_content(local_dir):

        log(
            f"SKIP {name} — already exists at "
            f"{local_dir}"
        )

        return True

    log("")
    log(f"Cloning baseline: {name}")
    log(f"  URL:    {git_url}")
    log(f"  Target: {local_dir}")

    ok = run_git_clone(
        git_url,
        local_dir,
        dry_run=dry_run,
    )

    if ok:

        log(
            f"  ✓ {name} cloned"
        )

    return ok


# ============================================================
# SAPLMA LOCAL BASELINE
# ============================================================

def create_saplma_stub(
    dry_run=False,
):
    """
    SAPLMA does not need to be treated as another target LLM.

    We implement a supervision-matched nonlinear hidden-state
    classifier locally so it uses the EXACT BRIDGE manifest
    and train/validation/test splits.
    """

    if dir_has_content(SAPLMA_DIR):

        log(
            f"SKIP saplma — already exists at "
            f"{SAPLMA_DIR}"
        )

        return True

    log("")
    log(
        "Creating SAPLMA-style local baseline directory"
    )

    if dry_run:

        log(
            f"  [DRY RUN] Would create "
            f"{SAPLMA_DIR}"
        )

        return True

    ensure_dir(SAPLMA_DIR)

    readme_path = os.path.join(
        SAPLMA_DIR,
        "README.md",
    )

    with open(
        readme_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            """# SAPLMA-style Supervised Hidden-State Baseline

Purpose
-------

Provide a supervision- and access-matched comparator for the
BRIDGE linear probe.

IMPORTANT:
This should use exactly the same:

- HealthBench manifest
- train split
- validation split
- test split
- target-model hidden states
- layer-selection protocol

Suggested experiment
--------------------

Input:
    last-token hidden state h_l

Classifier:
    small MLP, e.g.

        Linear(d, 256)
        ReLU
        Dropout
        Linear(256, 64)
        ReLU
        Linear(64, 1)

Compare against:
    BRIDGE logistic regression probe

Report:
    AUROC
    AUPRC
    Balanced Accuracy
    Sensitivity / Recall (Class 1)
    Specificity (Class 0)
    F1
    95% bootstrap CI

Do NOT use a different dataset split.

Reference motivation:
Azaria & Mitchell, EMNLP 2023.
"""
        )

    log(
        "  ✓ SAPLMA-style baseline stub created"
    )

    return True


def clone_all_baselines(
    name_filter=None,
    dry_run=False,
):
    """Clone all baseline repositories."""

    log("=" * 60)
    log("CLONING DETECTION BASELINES")
    log("=" * 60)

    ensure_dir(BASELINES_DIR)

    results = {}

    # --------------------------------------------------------
    # Local SAPLMA baseline
    # --------------------------------------------------------

    if name_filter == "saplma":

        ok = create_saplma_stub(
            dry_run=dry_run
        )

        results["saplma"] = (
            "success"
            if ok
            else "failed"
        )

        return results

    # --------------------------------------------------------
    # Git repositories
    # --------------------------------------------------------

    for name in BASELINE_SPECS:

        if (
            name_filter
            and name != name_filter
        ):
            continue

        ok = clone_baseline(
            name,
            dry_run=dry_run,
        )

        results[name] = (
            "success"
            if ok
            else "failed"
        )

    # --------------------------------------------------------
    # Always prepare our supervision-matched comparator
    # --------------------------------------------------------

    if not name_filter:

        ok = create_saplma_stub(
            dry_run=dry_run
        )

        results["saplma"] = (
            "success"
            if ok
            else "failed"
        )

    return results


# ============================================================
# MODEL VERIFICATION
# ============================================================

def get_nested_config_value(
    config,
    keys,
):
    """
    Search common nested model config dictionaries.

    Useful for multimodal models such as MedGemma / Lingshu,
    where text hidden-size information may live inside
    text_config / language_config / llm_config.
    """

    containers = [
        config,
        config.get("text_config", {}),
        config.get("language_config", {}),
        config.get("llm_config", {}),
    ]

    for container in containers:

        if not isinstance(container, dict):
            continue

        for key in keys:

            if key in container:
                return container[key]

    return "?"


def verify_models():
    """
    Verify that downloaded model repositories contain a
    config.json and print basic architecture information.
    """

    log("=" * 60)
    log("VERIFYING MODELS")
    log("=" * 60)

    for name in MODEL_SPECS:

        local_dir = model_dir(name)

        config_path = os.path.join(
            local_dir,
            "config.json",
        )

        if not os.path.exists(config_path):

            log(
                f"  ✗ {name}: config.json not found "
                f"at {local_dir}",
                "WARN",
            )

            continue

        try:

            with open(
                config_path,
                "r",
                encoding="utf-8",
            ) as f:

                config = json.load(f)

            hidden = get_nested_config_value(
                config,
                [
                    "hidden_size",
                    "d_model",
                ],
            )

            layers = get_nested_config_value(
                config,
                [
                    "num_hidden_layers",
                    "n_layer",
                    "num_layers",
                ],
            )

            mtype = config.get(
                "model_type",
                "?",
            )

            log(
                f"  ✓ {name:<15} "
                f"type={mtype}, "
                f"hidden={hidden}, "
                f"layers={layers}"
            )

        except Exception as e:

            log(
                f"  ✗ {name}: config read failed — "
                f"{e}",
                "ERROR",
            )


# ============================================================
# DATASET VERIFICATION
# ============================================================

def verify_healthbench(
    local_dir,
):
    """Verify the raw HealthBench files required by BRIDGE."""

    expected_files = [
        "2025-05-07-06-14-12_oss_eval.jsonl",
        "hard_2025-05-08-21-00-10.jsonl",
        "consensus_2025-05-09-20-00-46.jsonl",
    ]

    all_ok = True

    for filename in expected_files:

        path = os.path.join(
            local_dir,
            filename,
        )

        if os.path.exists(path):

            size_mb = (
                os.path.getsize(path)
                / (1024 ** 2)
            )

            log(
                f"    ✓ {filename} "
                f"({size_mb:.1f} MB)"
            )

        else:

            log(
                f"    ✗ Missing {filename}",
                "ERROR",
            )

            all_ok = False

    return all_ok


def verify_datasets():
    """Verify configured dataset downloads."""

    log("=" * 60)
    log("VERIFYING DATASETS")
    log("=" * 60)

    for name, spec in DATASET_SPECS.items():

        local_dir = dataset_dir(name)

        if not dir_has_content(local_dir):

            log(
                f"  ✗ {name}: not found at "
                f"{local_dir}",
                "WARN",
            )

            continue

        log(
            f"  ✓ {name}: found at {local_dir}"
        )

        if name == "healthbench":

            verify_healthbench(
                local_dir
            )

        elif (
            spec["source_type"]
            == "hf_dataset"
        ):

            try:

                from datasets import load_from_disk

                ds = load_from_disk(
                    local_dir
                )

                for (
                    split_name,
                    split_data,
                ) in ds.items():

                    log(
                        f"    ✓ {split_name}: "
                        f"{len(split_data)} examples"
                    )

            except Exception as e:

                log(
                    f"    ✗ Failed to load: {e}",
                    "ERROR",
                )

        elif (
            spec["source_type"]
            == "git"
        ):

            git_dir = os.path.join(
                local_dir,
                ".git",
            )

            if os.path.isdir(git_dir):

                log(
                    "    ✓ Git repository present"
                )

            else:

                log(
                    "    ⚠ Directory exists but "
                    "does not look like a Git clone",
                    "WARN",
                )


# ============================================================
# STATUS
# ============================================================

def print_status():
    """Print status of all configured resources."""

    log("=" * 60)
    log("DOWNLOAD STATUS")
    log("=" * 60)

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    log("")
    log("TARGET / EVALUATION MODELS:")

    for name, spec in MODEL_SPECS.items():

        local_dir = model_dir(name)

        icon = (
            "✓"
            if dir_has_content(local_dir)
            else "✗"
        )

        role = spec["role"]

        log(
            f"  {icon} "
            f"{name:<15} "
            f"[{role:<8}] "
            f"{local_dir}"
        )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    log("")
    log("DATASETS:")

    for name, spec in DATASET_SPECS.items():

        local_dir = dataset_dir(name)

        icon = (
            "✓"
            if dir_has_content(local_dir)
            else "✗"
        )

        optional = (
            "optional"
            if spec.get("optional", False)
            else "primary"
        )

        log(
            f"  {icon} "
            f"{name:<15} "
            f"[{optional:<8}] "
            f"{local_dir}"
        )

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    log("")
    log("DETECTION BASELINES:")

    for name in BASELINE_SPECS:

        local_dir = baseline_dir(name)

        icon = (
            "✓"
            if dir_has_content(local_dir)
            else "✗"
        )

        log(
            f"  {icon} "
            f"{name:<20} "
            f"{local_dir}"
        )

    saplma_icon = (
        "✓"
        if dir_has_content(SAPLMA_DIR)
        else "✗"
    )

    log(
        f"  {saplma_icon} "
        f"{'saplma':<20} "
        f"{SAPLMA_DIR}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    all_results,
):
    if not all_results:
        return

    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)

    all_ok = True

    for category, items in all_results.items():

        for name, status in items.items():

            icon = (
                "✓"
                if status == "success"
                else "✗"
            )

            log(
                f"  {icon} "
                f"{category}/{name}: "
                f"{status}"
            )

            if status != "success":
                all_ok = False

    if all_ok:

        log("")
        log(
            "All requested resources completed "
            "successfully!"
        )

    else:

        log("")
        log(
            "Some resources failed — check "
            "the errors above.",
            "WARN",
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "BRIDGE Revision — download target models, "
            "datasets, and detection baselines."
        )
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Download all PRIMARY revision resources"
        ),
    )

    parser.add_argument(
        "--models",
        action="store_true",
        help="Download target/evaluation models",
    )

    parser.add_argument(
        "--datasets",
        action="store_true",
        help="Download datasets",
    )

    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Clone baseline repositories",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded resources",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Print download status",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Download only the specified resource name"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show what would happen without "
            "downloading anything"
        ),
    )

    parser.add_argument(
        "--include-optional",
        action="store_true",
        help=(
            "Also download optional datasets "
            "(MediQ and PubMedQA)"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Default action
    # --------------------------------------------------------

    if not any(
        [
            args.all,
            args.models,
            args.datasets,
            args.baselines,
            args.verify,
            args.status,
        ]
    ):

        args.status = True

    # --------------------------------------------------------
    # Status only
    # --------------------------------------------------------

    if args.status:

        print_status()
        return

    # --------------------------------------------------------
    # HuggingFace login check
    # --------------------------------------------------------

    if (
        args.all
        or args.models
        or args.datasets
    ):

        logged_in = check_hf_login()

        if (
            not logged_in
            and not args.dry_run
        ):

            log(
                "Continuing anyway. Public resources "
                "may still download, but MedGemma "
                "requires authenticated access.",
                "WARN",
            )

    all_results = {}

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    if (
        args.all
        or args.models
    ):

        all_results["models"] = (
            download_all_models(
                name_filter=args.name,
                dry_run=args.dry_run,
            )
        )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    if (
        args.all
        or args.datasets
    ):

        all_results["datasets"] = (
            download_all_datasets(
                name_filter=args.name,
                dry_run=args.dry_run,
                include_optional=(
                    args.include_optional
                ),
            )
        )

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    if (
        args.all
        or args.baselines
    ):

        all_results["baselines"] = (
            clone_all_baselines(
                name_filter=args.name,
                dry_run=args.dry_run,
            )
        )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    if args.verify:

        verify_models()
        verify_datasets()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_summary(
        all_results
    )


if __name__ == "__main__":
    main()