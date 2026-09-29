#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRIDGE causal intervention experiments.

Experiments
-----------
1. Directional causal use at the selected probe layer
   - add +/- alpha * sigma_c * CAV
   - matched orthogonal random-direction controls

2. Layer sweep: where does the insufficiency signal causally matter?
   - IMPORTANT: uses a DIFFERENT locally trained CAV c_l at every layer l
   - layer-wise CAVs are reconstructed from the cached TRAIN activations
     with the same logistic-regression settings as BRIDGE
   - the reconstructed selected-layer CAV is checked against the saved
     BRIDGE selected CAV as a sanity check
   - saves validation AUROC for each layer (from layer_validation.csv) so
     causal-effect-vs-decodability plots can be made later

3. Necessity via mean-centered projection ablation
   - removes the naturally occurring sample-specific component along the
     selected-layer CAV
   - matched orthogonal random-direction ablation controls

Key methodological choices
--------------------------
* Development/tuning should use the frozen HealthBench VALIDATION set only.
* HealthBench Hard must not be used to choose alpha/layer/intervention.
* Intervention is PREFILL-ONLY and LAST-PROMPT-TOKEN only.
* Greedy decoding is used by default for clean paired causal comparisons.
* All layer indices used internally are 0-based.
* The selected layer is read automatically from outputs/probes/summary.csv.
* The saved selected CAV is read from outputs/probes/<model>/cav.npy.
* Experiment 2 does NOT transport the selected-layer CAV across layers.
  It reconstructs one probe/CAV per layer from TRAIN activations.

Expected BRIDGE artifacts
-------------------------
outputs/probes/summary.csv
outputs/probes/<model>/cav.npy
outputs/probes/<model>/layer_validation.csv

Expected activation cache
-------------------------
cache/activations/<model>/healthbench/train/*.pt
(or pass the exact TRAIN activation directory with --train-activation-dir).

The .pt files may contain:
  activations / hidden_states / last_token_hidden_states / layer_activations
and ideally label + example_id. If label is absent, --manifest is used.

Typical commands
----------------
# Expt 1
python scripts/causal_interventions.py \
  --experiment 1 \
  --model-key biomistral \
  --model-path models/biomistral \
  --input-jsonl data/healthbench/2025-05-07-06-14-12_oss_eval.jsonl \
  --train-activation-dir cache/activations/biomistral/healthbench/train \
  --manifest data/manifests/healthbench_evidence_sufficiency_v1.csv \
  --probe-root outputs/probes \
  --output-dir outputs/causal/biomistral/expt1_pilot \
  --max-per-class 25 \
  --alphas=-2,-1,-0.5,0.5,1,2

# Expt 2
python scripts/causal_interventions.py \
  --experiment 2 \
  --model-key biomistral \
  --model-path models/biomistral \
  --input-jsonl data/healthbench/2025-05-07-06-14-12_oss_eval.jsonl \
  --train-activation-dir cache/activations/biomistral/healthbench/train \
  --manifest data/manifests/healthbench_evidence_sufficiency_v1.csv \
  --probe-root outputs/probes \
  --output-dir outputs/causal/biomistral/expt2_pilot \
  --max-per-class 15 \
  --layers 12,14,16,18,20 \
  --alphas=-1,1 \
  --include-random

# Expt 3
python scripts/causal_interventions.py \
  --experiment 3 \
  --model-key biomistral \
  --model-path models/biomistral \
  --input-jsonl data/healthbench/2025-05-07-06-14-12_oss_eval.jsonl \
  --train-activation-dir cache/activations/biomistral/healthbench/train \
  --manifest data/manifests/healthbench_evidence_sufficiency_v1.csv \
  --probe-root outputs/probes \
  --output-dir outputs/causal/biomistral/expt3_pilot \
  --max-per-class 25 \
  --lambdas=0.5,1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSUFFICIENT_TAGS = {
    "physician_agreed_category:not-enough-context",
    "physician_agreed_category:not-enough-info-to-complete-task",
}
SUFFICIENT_TAGS = {
    "physician_agreed_category:enough-context",
    "physician_agreed_category:enough-info-to-complete-task",
}

PROBE_C = 1.0
PROBE_SEED = 42
PROBE_MAX_ITER = 10000

INSUFFICIENT_TAGS = {
    "physician_agreed_category:not-enough-context",
    "physician_agreed_category:not-enough-info-to-complete-task",
}

SUFFICIENT_TAGS = {
    "physician_agreed_category:enough-context",
    "physician_agreed_category:enough-info-to-complete-task",
}


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at {path}:{line_no}: {e}") from e
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# BRIDGE probe artifacts
# ---------------------------------------------------------------------------

def load_selected_layer(probe_root: Path, model_key: str) -> tuple[int, int]:
    """
    Returns:
        selected_layer_0based, selected_layer_1based

    We deliberately read the canonical outputs/probes/summary.csv rather than
    hard-coding layer numbers.  The hook MUST use the 0-based index.
    """
    summary_path = probe_root / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Probe summary not found: {summary_path}")

    df = pd.read_csv(summary_path)
    rows = df[df["model"].astype(str) == str(model_key)].copy()
    if len(rows) == 0:
        raise KeyError(f"{model_key} not found in {summary_path}")

    if "selected_layer_0based" not in rows.columns:
        raise KeyError(
            f"{summary_path} lacks selected_layer_0based. "
            "Do not infer the hook index from a 1-based paper layer."
        )

    zero = sorted(set(rows["selected_layer_0based"].dropna().astype(int)))
    if len(zero) != 1:
        raise RuntimeError(
            f"Expected one frozen selected layer for {model_key}; found {zero}"
        )

    selected0 = zero[0]

    if "selected_layer_1based" in rows.columns:
        one = sorted(set(rows["selected_layer_1based"].dropna().astype(int)))
        selected1 = one[0] if len(one) == 1 else selected0 + 1
    else:
        selected1 = selected0 + 1

    if selected1 != selected0 + 1:
        raise RuntimeError(
            f"Inconsistent layer numbering: 0-based={selected0}, 1-based={selected1}"
        )

    return selected0, selected1


def load_layer_validation_auc(probe_root: Path, model_key: str) -> dict[int, float]:
    path = probe_root / model_key / "layer_validation.csv"
    if not path.exists():
        print(f"[WARN] {path} missing; causal rows will omit probe_val_auroc.")
        return {}

    df = pd.read_csv(path)

    layer_col = None
    for c in ("layer_index_0based", "layer_0based", "layer", "layer_idx", "layer_index"):
        if c in df.columns:
            layer_col = c
            break

    auc_col = None
    for c in ("auroc", "val_auroc", "validation_auroc"):
        if c in df.columns:
            auc_col = c
            break

    if layer_col is None or auc_col is None:
        print(
            f"[WARN] Could not identify layer/AUROC columns in {path}: "
            f"{list(df.columns)}"
        )
        return {}

    return {
        int(row[layer_col]): float(row[auc_col])
        for _, row in df.iterrows()
        if pd.notna(row[layer_col]) and pd.notna(row[auc_col])
    }


def load_selected_cav(probe_root: Path, model_key: str) -> torch.Tensor:
    path = probe_root / model_key / "cav.npy"
    if not path.exists():
        raise FileNotFoundError(f"Selected CAV missing: {path}")

    cav = torch.as_tensor(np.load(path), dtype=torch.float32).reshape(-1)
    norm = cav.norm().item()
    if norm <= 0:
        raise ValueError(f"Zero-norm CAV: {path}")
    cav = cav / norm
    print(f"Selected CAV: {path} | dim={cav.numel()} | raw_norm={norm:.6g}")
    return cav


# ---------------------------------------------------------------------------
# HealthBench examples / frozen validation split
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: tuple[str, ...], required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Could not find any of {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .isin({"true", "1", "yes", "y"})
    )


def load_manifest(manifest_path: Path):
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")

    df = pd.read_csv(manifest_path)

    cols = {
        "split": _find_column(df, ("split", "dataset_split", "partition")),
        "id": _find_column(
            df, ("example_id", "prompt_id", "source_prompt_id", "id"), required=False
        ),
        "label": _find_column(
            df, ("binary_label", "label", "evidence_label", "y")
        ),
        "prompt": _find_column(
            df, ("prompt", "prompt_text", "text"), required=False
        ),
        "included": _find_column(
            df, ("included", "is_included"), required=False
        ),
        "source_line": _find_column(
            df, ("source_line", "line_number", "source_row"), required=False
        ),
        "source_file": _find_column(
            df, ("source_file", "source", "file"), required=False
        ),
    }

    if cols["included"] is not None:
        mask = _truthy(df[cols["included"]])
        if mask.any():
            df = df[mask].copy()

    return df, cols


def manifest_split(df: pd.DataFrame, cols: dict, split_name: str) -> pd.DataFrame:
    split = (
        df[cols["split"]]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    aliases = {
        "train": {"train", "training"},
        "validation": {"val", "validation", "dev"},
        "test": {"test", "hard", "healthbench_hard"},
    }

    out = df[split.isin(aliases.get(split_name, {split_name}))].copy()
    labels = pd.to_numeric(out[cols["label"]], errors="coerce")
    out = out[labels.isin([0, 1])].copy()
    return out


def normalize_match_text(value: Any) -> str:
    value = str(value)
    value = value.replace("\\r\\n", "\\n").replace("\\r", "\\n")
    value = re.sub(r"\\s+", " ", value).strip().lower()
    return value


def prompt_signatures(prompt: Any) -> set[str]:
    sigs = set()

    if isinstance(prompt, str):
        sigs.add(normalize_match_text(prompt))

    try:
        messages = normalize_messages(prompt)
    except Exception:
        messages = []

    if messages:
        joined = "\\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        sigs.add(normalize_match_text(joined))

        contents = "\\n".join(m["content"] for m in messages)
        sigs.add(normalize_match_text(contents))

        for m in messages:
            sigs.add(normalize_match_text(m["content"]))

        user_msgs = [
            m["content"] for m in messages if m["role"].lower() == "user"
        ]
        if user_msgs:
            sigs.add(normalize_match_text(user_msgs[-1]))

        sigs.add(normalize_match_text(messages[-1]["content"]))

    return {s for s in sigs if s}


def infer_healthbench_target_label(raw_row: dict) -> int | None:
    """
    Infer the strict BRIDGE evidence-sufficiency label from canonical
    HealthBench physician-agreed target tags.
    """
    tags = set(raw_row.get("example_tags", []) or [])

    has_insufficient = bool(tags & INSUFFICIENT_TAGS)
    has_sufficient = bool(tags & SUFFICIENT_TAGS)

    if has_insufficient and has_sufficient:
        raise RuntimeError(
            "A raw HealthBench example contains both sufficient and "
            "insufficient target tags."
        )

    if has_insufficient:
        return 1

    if has_sufficient:
        return 0

    return None


def load_eval_examples(
    input_jsonl: Path,
    manifest: Path,
    max_per_class: int | None,
    seed: int,
) -> list[dict]:
    """
    Recover the exact frozen 123-example validation split.

    BRIDGE manifest IDs such as oss_eval_000030_2216d65f are synthetic
    identifiers and are not the UUID prompt_id values in raw HealthBench.

    The immutable manifest already stores source_line. We therefore map each
    validation row back to the raw OSS-Eval JSONL by source_line.

    The code automatically checks whether source_line is one-based or
    zero-based. Every mapped row must reproduce the manifest binary label from
    the raw physician-agreed tags, preventing a silent off-by-one mapping.
    """
    manifest_df, cols = load_manifest(manifest)
    val = manifest_split(manifest_df, cols, "validation")

    if len(val) != 123:
        raise RuntimeError(
            f"Expected 123 validation rows in the manifest, found {len(val)}."
        )

    if cols.get("source_line") is None:
        raise RuntimeError(
            "Manifest does not contain source_line. Exact mapping from the "
            "BRIDGE synthetic example IDs to raw OSS-Eval is therefore "
            "unavailable."
        )

    raw_rows = read_jsonl(input_jsonl)

    # Test both plausible conventions:
    #   offset -1: source_line is one-based
    #   offset  0: source_line is zero-based
    offset_scores = {}

    for offset in (-1, 0):
        matches = 0
        in_range = 0

        for _, mrow in val.iterrows():
            try:
                source_line = int(float(mrow[cols["source_line"]]))
            except Exception:
                continue

            raw_index = source_line + offset

            if not (0 <= raw_index < len(raw_rows)):
                continue

            in_range += 1

            manifest_label = int(float(mrow[cols["label"]]))
            raw_label = infer_healthbench_target_label(raw_rows[raw_index])

            if raw_label == manifest_label:
                matches += 1

        offset_scores[offset] = {
            "matches": matches,
            "in_range": in_range,
        }

    print(
        "source_line mapping scores: "
        f"one_based={offset_scores[-1]['matches']}/123, "
        f"zero_based={offset_scores[0]['matches']}/123"
    )

    best_offset = max(
        offset_scores,
        key=lambda offset: offset_scores[offset]["matches"],
    )

    best_matches = offset_scores[best_offset]["matches"]

    if best_matches != 123:
        raise RuntimeError(
            "Could not establish an exact source_line mapping from the "
            "validation manifest to OSS-Eval. "
            f"Best convention matched {best_matches}/123 labels. "
            f"Scores: {offset_scores}"
        )

    if best_offset == -1:
        print(
            "source_line convention: one-based "
            "(raw JSONL index = source_line - 1)"
        )
    else:
        print(
            "source_line convention: zero-based "
            "(raw JSONL index = source_line)"
        )

    mapped = []

    for _, mrow in val.iterrows():
        source_line = int(float(mrow[cols["source_line"]]))
        raw_index = source_line + best_offset

        if not (0 <= raw_index < len(raw_rows)):
            raise RuntimeError(
                f"source_line={source_line} maps outside OSS-Eval."
            )

        raw = raw_rows[raw_index]
        manifest_label = int(float(mrow[cols["label"]]))
        raw_label = infer_healthbench_target_label(raw)

        if raw_label != manifest_label:
            raise RuntimeError(
                "source_line mapping label mismatch: "
                f"source_line={source_line}, raw_index={raw_index}, "
                f"manifest_label={manifest_label}, raw_label={raw_label}, "
                f"tags={raw.get('example_tags', [])}"
            )

        prompt_id = raw.get(
            "prompt_id",
            raw.get("example_id", raw.get("id")),
        )

        if prompt_id is None:
            raise RuntimeError(
                f"OSS-Eval row {raw_index} has no canonical prompt_id."
            )

        prompt = raw.get(
            "prompt",
            raw.get("messages", raw.get("conversation")),
        )

        if prompt is None:
            raise RuntimeError(
                f"OSS-Eval prompt_id={prompt_id} has no prompt."
            )

        manifest_id = None

        if cols.get("id") is not None and pd.notna(mrow[cols["id"]]):
            manifest_id = str(mrow[cols["id"]])

        source_file = None

        if (
            cols.get("source_file") is not None
            and pd.notna(mrow[cols["source_file"]])
        ):
            source_file = str(mrow[cols["source_file"]])

        mapped.append(
            {
                "prompt_id": str(prompt_id),
                "manifest_id": manifest_id,
                "source_file": source_file,
                "source_line": source_line,
                "raw_index": raw_index,
                "prompt": prompt,
                "label": manifest_label,
                "example_tags": raw.get("example_tags", []),
            }
        )

    if len(mapped) != 123:
        raise RuntimeError(
            f"Expected 123 mapped validation examples, got {len(mapped)}."
        )

    if len({example["prompt_id"] for example in mapped}) != 123:
        raise RuntimeError(
            "Validation mapping is not one-to-one with canonical "
            "HealthBench prompt IDs."
        )

    class0 = sum(example["label"] == 0 for example in mapped)
    class1 = sum(example["label"] == 1 for example in mapped)

    if class0 != 78 or class1 != 45:
        raise RuntimeError(
            f"Validation class counts changed after mapping: "
            f"class0={class0}, class1={class1}; expected 78/45."
        )

    print(
        "Validation source mapping: PASS "
        f"(123 examples; class0={class0}, class1={class1})"
    )

    if max_per_class is not None:
        rng = random.Random(seed)
        selected = []

        for label in (0, 1):
            group = [
                example
                for example in mapped
                if example["label"] == label
            ]

            rng.shuffle(group)
            selected.extend(
                group[: min(max_per_class, len(group))]
            )

        rng.shuffle(selected)
        mapped = selected

    print(
        f"Evaluation examples: {len(mapped)} "
        f"(sufficient={sum(x['label'] == 0 for x in mapped)}, "
        f"insufficient={sum(x['label'] == 1 for x in mapped)})"
    )

    return mapped


# ---------------------------------------------------------------------------
# Model loading / formatting
# ---------------------------------------------------------------------------

def load_model_interface(model_path: str):
    """
    Load the model onto the single CUDA device exposed by SLURM.

    The scheduler-assigned physical GPU should appear as logical cuda:0 via
    CUDA_VISIBLE_DEVICES. We test the device before loading model weights so a
    bad/busy allocation fails immediately and clearly.
    """
    import os

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoProcessor,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable inside this job. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
            f"SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS')}"
        )

    if torch.cuda.device_count() < 1:
        raise RuntimeError(
            "PyTorch sees zero CUDA devices. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
            f"SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS')}"
        )

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    print(
        "CUDA allocation: "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
        f"SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS')}, "
        f"visible_device_count={torch.cuda.device_count()}, "
        f"device_name={torch.cuda.get_device_name(0)}"
    )

    try:
        test_tensor = torch.zeros(1, device=device)
        torch.cuda.synchronize()
        del test_tensor
    except RuntimeError as exc:
        raise RuntimeError(
            "The scheduler-assigned GPU is visible but cannot be used. "
            "This is a GPU allocation/node availability problem. "
            "Resubmit the job on a free GPU/node."
        ) from exc

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    archs = list(getattr(config, "architectures", []) or [])
    arch = archs[0] if archs else ""

    dtype = torch.bfloat16
    processor = None

    if "Qwen2_5_VLForConditionalGeneration" in arch:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        tokenizer = processor.tokenizer
        kind = "qwen2_5_vl"

    elif "Gemma3ForConditionalGeneration" in arch:
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            tokenizer = processor.tokenizer
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

        kind = "gemma3"

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        kind = "causal_lm"

    try:
        model.to(device)
    except RuntimeError as exc:
        raise RuntimeError(
            "Model weights loaded from disk, but moving the model to the "
            "SLURM-assigned GPU failed. This normally means the allocated GPU "
            "became unavailable or is incorrectly shared. Resubmit the job."
        ) from exc

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model class:     {model.__class__.__name__}")
    print(f"Config class:    {config.__class__.__name__}")
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Model kind:      {kind}")

    if processor is not None:
        print(f"Processor class: {processor.__class__.__name__}")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "processor": processor,
        "config": config,
        "device": device,
        "kind": kind,
    }


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content)


def normalize_messages(prompt: Any) -> list[dict]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]

    if isinstance(prompt, dict):
        if "role" in prompt and "content" in prompt:
            return [
                {
                    "role": str(prompt["role"]),
                    "content": _content_to_text(prompt["content"]),
                }
            ]
        return [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]

    if isinstance(prompt, list):
        msgs = []
        for item in prompt:
            if isinstance(item, dict) and "role" in item and "content" in item:
                msgs.append(
                    {
                        "role": str(item["role"]),
                        "content": _content_to_text(item["content"]),
                    }
                )
        if msgs:
            return msgs

    raise ValueError(f"Unsupported prompt format: {type(prompt)}")


def llama3_manual_template(messages: list[dict]) -> str:
    pieces = ["<|begin_of_text|>"]
    for msg in messages:
        pieces.append(
            f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
            f"{msg['content']}<|eot_id|>"
        )
    pieces.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(pieces)


def prepare_model_inputs(bundle: dict, prompt: Any, model_key: str) -> dict:
    tokenizer = bundle["tokenizer"]
    processor = bundle["processor"]
    device = bundle["device"]
    kind = bundle["kind"]
    messages = normalize_messages(prompt)

    if kind == "qwen2_5_vl":
        # Text-only Qwen2.5-VL: use the processor, not a generic tokenizer path.
        formatted = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = processor(
            text=[formatted],
            return_tensors="pt",
            padding=True,
        )

    elif getattr(tokenizer, "chat_template", None):
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if processor is not None and kind == "gemma3":
            # Gemma3 processor can safely handle text-only input.
            enc = processor(
                text=[formatted],
                return_tensors="pt",
                padding=True,
            )
        else:
            enc = tokenizer(
                formatted,
                return_tensors="pt",
                add_special_tokens=False,
            )

    elif "openbio" in model_key.lower():
        formatted = llama3_manual_template(messages)
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=False,
        )

    else:
        formatted = "".join(
            f"{m['role'].upper()}: {m['content']}\n" for m in messages
        ) + "ASSISTANT:"
        enc = tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=False,
        )

    # Keep all tensor fields because Qwen/Gemma processors may emit fields
    # beyond input_ids/attention_mask that their model accepts.
    enc = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in enc.items()
    }
    return enc


# ---------------------------------------------------------------------------
# Locate transformer layers
# ---------------------------------------------------------------------------

def _get_by_path(obj: Any, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def infer_num_hidden_layers(config) -> int | None:
    for obj in (config, getattr(config, "text_config", None)):
        if obj is None:
            continue
        for key in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(obj, key, None)
            if isinstance(value, int):
                return int(value)
    return None


def infer_hidden_size(config) -> int | None:
    for obj in (config, getattr(config, "text_config", None)):
        if obj is None:
            continue
        value = getattr(obj, "hidden_size", None)
        if isinstance(value, int):
            return int(value)
    return None


def resolve_decoder_layers(model: nn.Module, config):
    expected = infer_num_hidden_layers(config)

    paths = [
        "model.layers",
        "model.model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "model.language_model.layers",
        "model.language_model.model.layers",
    ]

    for path in paths:
        layers = _get_by_path(model, path)
        if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
            if expected is None or len(layers) == expected:
                print(f"Decoder layers: {path} ({len(layers)} layers)")
                return layers

    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            first = module[0]
            if hasattr(first, "self_attn") or hasattr(first, "attention"):
                score = len(module)
                if expected is not None and len(module) == expected:
                    score += 10000
                candidates.append((score, name, module))

    if not candidates:
        raise RuntimeError("Could not locate decoder transformer blocks.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, name, layers = candidates[0]
    print(f"Decoder layers by search: {name} ({len(layers)} layers)")
    return layers


# ---------------------------------------------------------------------------
# Activation cache loading
# ---------------------------------------------------------------------------

def extract_activation_matrix(obj: Any, hidden_dim: int) -> torch.Tensor:
    candidates = []

    if torch.is_tensor(obj):
        candidates.append(obj)
    elif isinstance(obj, dict):
        for key in (
            "activations",
            "hidden_states",
            "last_token_hidden_states",
            "layer_activations",
            "h",
        ):
            if key in obj and torch.is_tensor(obj[key]):
                candidates.append(obj[key])
        # fallback: any tensor
        for value in obj.values():
            if torch.is_tensor(value):
                candidates.append(value)

    for t in candidates:
        t = t.detach().float().cpu()

        if t.ndim == 2 and t.shape[-1] == hidden_dim:
            return t

        if t.ndim == 3 and t.shape[0] == 1 and t.shape[-1] == hidden_dim:
            t2 = t.squeeze(0)
            if t2.ndim == 2:
                return t2

    raise ValueError(
        f"Could not extract [layers, {hidden_dim}] activation matrix."
    )


def extract_cache_id(obj: Any, path: Path) -> str:
    if isinstance(obj, dict):
        for key in ("example_id", "prompt_id", "id"):
            if key in obj and obj[key] is not None:
                return str(obj[key])
    return path.stem


def extract_cache_label(obj: Any) -> int | None:
    if isinstance(obj, dict):
        for key in ("label", "binary_label", "evidence_label", "y"):
            if key in obj and obj[key] is not None:
                try:
                    return int(float(obj[key]))
                except Exception:
                    pass
    return None


def load_manifest_label_map(manifest: Path | None) -> dict[str, int]:
    if manifest is None:
        return {}
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest missing: {manifest}")

    df = pd.read_csv(manifest)

    id_col = None
    for c in ("example_id", "prompt_id", "id"):
        if c in df.columns:
            id_col = c
            break

    label_col = None
    for c in ("binary_label", "label", "evidence_label"):
        if c in df.columns:
            label_col = c
            break

    if id_col is None or label_col is None:
        raise RuntimeError(
            f"Cannot map labels from {manifest}; columns={list(df.columns)}"
        )

    if "split" in df.columns:
        # Only training labels are relevant for reconstructing the probes.
        split = df["split"].astype(str).str.lower().str.strip()
        df = df[split == "train"].copy()

    out = {}
    for _, row in df.iterrows():
        if pd.isna(row[id_col]) or pd.isna(row[label_col]):
            continue
        try:
            label = int(float(row[label_col]))
        except Exception:
            continue
        if label in (0, 1):
            out[str(row[id_col])] = label

    return out


def load_training_activation_tensor(
    cache_dir: Path,
    hidden_dim: int,
    n_layers: int,
    manifest: Path | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
      X: [N, L, D]
      y: [N]
      ids: N strings
    """
    files = sorted(cache_dir.rglob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files under {cache_dir}")

    manifest_map = load_manifest_label_map(manifest)

    xs, ys, ids = [], [], []

    for path in files:
        obj = torch.load(path, map_location="cpu", weights_only=False)

        matrix = extract_activation_matrix(obj, hidden_dim)
        if matrix.shape[0] < n_layers:
            raise RuntimeError(
                f"{path}: only {matrix.shape[0]} layers, expected >= {n_layers}"
            )
        matrix = matrix[:n_layers]

        example_id = extract_cache_id(obj, path)
        label = extract_cache_label(obj)

        if label is None:
            label = manifest_map.get(example_id)

        if label not in (0, 1):
            raise RuntimeError(
                f"Could not resolve binary label for cache file {path} "
                f"(example_id={example_id})."
            )

        xs.append(matrix.numpy())
        ys.append(label)
        ids.append(example_id)

    X = np.stack(xs, axis=0).astype(np.float32, copy=False)
    y = np.asarray(ys, dtype=np.int64)

    print(
        f"TRAIN cache loaded: X={X.shape}, "
        f"class0={(y==0).sum()}, class1={(y==1).sum()}"
    )

    if len(X) != 492:
        print(
            f"[WARN] Expected 492 HealthBench training examples, found {len(X)}. "
            "Proceed only if this is intentional."
        )

    return X, y, ids


# ---------------------------------------------------------------------------
# Reconstruct local CAV at every layer
# ---------------------------------------------------------------------------

def build_or_load_layerwise_cavs(
    probe_root: Path,
    model_key: str,
    train_activation_dir: Path,
    manifest: Path | None,
    n_layers: int,
    hidden_dim: int,
    selected_layer: int,
    saved_selected_cav: torch.Tensor,
    force_rebuild: bool,
) -> dict[str, np.ndarray]:
    """
    Existing BRIDGE outputs preserve only the final selected probe weights.
    The original training log confirms that a separate logistic probe was
    trained at every layer, but those weights were not saved.

    For Expt 2 we reconstruct those SAME train-only layer probes:
      LogisticRegression(
          penalty='l2', C=1.0, class_weight='balanced',
          solver='liblinear', random_state=42, max_iter=10000
      )

    This uses cached activations only; no LLM/GPU work is needed.
    """
    bank_path = probe_root / model_key / "layerwise_cavs.npz"

    if bank_path.exists() and not force_rebuild:
        z = np.load(bank_path)
        needed = {"cavs", "weights", "intercepts", "projection_stds"}
        if needed.issubset(set(z.files)):
            cavs = z["cavs"]
            if cavs.shape == (n_layers, hidden_dim):
                print(f"Layer-wise CAV bank loaded: {bank_path}")
                return {k: z[k] for k in z.files}

        print(f"[WARN] Existing {bank_path} is incompatible; rebuilding.")

    X, y, _ = load_training_activation_tensor(
        cache_dir=train_activation_dir,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        manifest=manifest,
    )

    weights = np.zeros((n_layers, hidden_dim), dtype=np.float32)
    cavs = np.zeros((n_layers, hidden_dim), dtype=np.float32)
    intercepts = np.zeros(n_layers, dtype=np.float32)
    projection_stds = np.zeros(n_layers, dtype=np.float32)

    print("Reconstructing one train-only logistic probe per layer...")

    for layer in range(n_layers):
        clf = LogisticRegression(
            penalty="l2",
            C=PROBE_C,
            class_weight="balanced",
            solver="liblinear",
            random_state=PROBE_SEED,
            max_iter=PROBE_MAX_ITER,
        )
        clf.fit(X[:, layer, :], y)

        w = clf.coef_.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(w))
        if norm <= 0:
            raise RuntimeError(f"Layer {layer}: zero-norm probe weight.")

        c = w / norm
        projections = X[:, layer, :] @ c
        sigma = float(np.std(projections, ddof=1))
        if not np.isfinite(sigma) or sigma <= 1e-8:
            raise RuntimeError(f"Layer {layer}: invalid projection std {sigma}")

        weights[layer] = w
        cavs[layer] = c
        intercepts[layer] = float(clf.intercept_[0])
        projection_stds[layer] = sigma

        print(
            f"  layer {layer:2d}: "
            f"|w|={norm:.5g}, projection_std={sigma:.5g}"
        )

    # Critical sanity check: locally reconstructed selected-layer direction
    # should agree with the canonical saved selected CAV.
    reconstructed = torch.from_numpy(cavs[selected_layer]).float()
    cosine = float(torch.dot(reconstructed, saved_selected_cav).item())

    print(
        f"Selected-layer CAV reconstruction cosine "
        f"(layer {selected_layer}): {cosine:.8f}"
    )

    if cosine < 0.99:
        raise RuntimeError(
            "Reconstructed selected-layer CAV does not match the canonical "
            f"saved CAV (cosine={cosine:.6f}). This usually means cache/label "
            "alignment or probe settings differ. Do NOT run Expt 2 until fixed."
        )

    bank_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        bank_path,
        cavs=cavs,
        weights=weights,
        intercepts=intercepts,
        projection_stds=projection_stds,
        selected_layer_0based=np.asarray([selected_layer], dtype=np.int64),
        selected_cav_cosine=np.asarray([cosine], dtype=np.float32),
    )
    print(f"Saved layer-wise CAV bank: {bank_path}")

    return {
        "cavs": cavs,
        "weights": weights,
        "intercepts": intercepts,
        "projection_stds": projection_stds,
        "selected_layer_0based": np.asarray([selected_layer], dtype=np.int64),
        "selected_cav_cosine": np.asarray([cosine], dtype=np.float32),
    }


def compute_selected_projection_std(
    train_activation_dir: Path,
    manifest: Path | None,
    selected_cav: torch.Tensor,
    selected_layer: int,
    n_layers: int,
) -> float:
    X, _, _ = load_training_activation_tensor(
        train_activation_dir,
        hidden_dim=selected_cav.numel(),
        n_layers=n_layers,
        manifest=manifest,
    )
    c = selected_cav.numpy()
    proj = X[:, selected_layer, :] @ c
    sigma = float(np.std(proj, ddof=1))
    if not np.isfinite(sigma) or sigma <= 1e-8:
        raise RuntimeError(f"Invalid selected-layer projection std: {sigma}")
    return sigma


def compute_train_mean_at_layer(
    train_activation_dir: Path,
    manifest: Path | None,
    hidden_dim: int,
    n_layers: int,
    layer: int,
) -> torch.Tensor:
    X, _, _ = load_training_activation_tensor(
        train_activation_dir,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        manifest=manifest,
    )
    return torch.from_numpy(X[:, layer, :].mean(axis=0)).float()


# ---------------------------------------------------------------------------
# Random controls
# ---------------------------------------------------------------------------

def matched_random_direction(cav: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    r = torch.randn(cav.shape, generator=g, dtype=torch.float32)
    r = r - torch.dot(r, cav) * cav
    norm = r.norm()
    if norm < 1e-8:
        raise RuntimeError("Random direction collapsed during orthogonalization.")
    return r / norm


# ---------------------------------------------------------------------------
# Residual-stream hook
# ---------------------------------------------------------------------------

@dataclass
class InterventionRecord:
    fired: bool = False
    activation_norm: float | None = None
    cav_projection_before: float | None = None
    cav_projection_after: float | None = None
    intervention_projection_before: float | None = None
    intervention_projection_after: float | None = None
    delta_norm: float | None = None
    centered_projection_before: float | None = None
    centered_projection_after: float | None = None


def get_hidden_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported transformer block output: {type(output)}")


def replace_hidden_in_output(output: Any, hidden_new: torch.Tensor):
    if torch.is_tensor(output):
        return hidden_new
    if isinstance(output, tuple):
        return (hidden_new,) + tuple(output[1:])
    if isinstance(output, list):
        return [hidden_new] + list(output[1:])
    raise TypeError(f"Unsupported transformer block output: {type(output)}")


def make_prefill_last_token_hook(
    mode: str,
    intervention_direction: torch.Tensor,
    cav_direction: torch.Tensor,
    magnitude: float,
    record: InterventionRecord,
    train_mean: torch.Tensor | None = None,
):
    """
    Fires ONCE: the first model forward pass inside generate() (prompt prefill).
    Modifies hidden[:, -1, :] only.
    """
    if mode not in ("add", "ablate"):
        raise ValueError(mode)

    def hook(_module, _inputs, output):
        if record.fired:
            return output

        hidden = get_hidden_from_output(output)
        if hidden.ndim != 3:
            raise RuntimeError(
                f"Expected block output [batch, seq, hidden], got {tuple(hidden.shape)}"
            )

        d = intervention_direction.to(hidden.device, hidden.dtype)
        c = cav_direction.to(hidden.device, hidden.dtype)

        h = hidden[:, -1, :]
        h0 = h.detach().float()

        record.activation_norm = float(h0.norm(dim=-1).mean().item())
        record.cav_projection_before = float(
            (h0 * c.float()).sum(dim=-1).mean().item()
        )
        record.intervention_projection_before = float(
            (h0 * d.float()).sum(dim=-1).mean().item()
        )

        if mode == "add":
            h_new = h + float(magnitude) * d

        else:
            if train_mean is None:
                raise ValueError("Projection ablation requires train_mean.")
            mu = train_mean.to(hidden.device, hidden.dtype).unsqueeze(0)
            centered = h - mu
            coeff = (centered * d).sum(dim=-1, keepdim=True)

            record.centered_projection_before = float(
                coeff.detach().float().mean().item()
            )

            h_new = h - float(magnitude) * coeff * d

            coeff_after = ((h_new - mu) * d).sum(dim=-1, keepdim=True)
            record.centered_projection_after = float(
                coeff_after.detach().float().mean().item()
            )

        hidden_new = hidden.clone()
        hidden_new[:, -1, :] = h_new

        h1 = h_new.detach().float()
        record.cav_projection_after = float(
            (h1 * c.float()).sum(dim=-1).mean().item()
        )
        record.intervention_projection_after = float(
            (h1 * d.float()).sum(dim=-1).mean().item()
        )
        record.delta_norm = float((h1 - h0).norm(dim=-1).mean().item())
        record.fired = True

        return replace_hidden_in_output(output, hidden_new)

    return hook


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_one(
    bundle: dict,
    model_key: str,
    prompt: Any,
    layer_module: nn.Module | None,
    hook_fn,
    max_new_tokens: int,
    seed: int,
) -> tuple[str, int]:
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]

    enc = prepare_model_inputs(bundle, prompt, model_key)
    if "input_ids" not in enc:
        raise RuntimeError("Prepared model inputs do not contain input_ids.")

    prompt_len = int(enc["input_ids"].shape[1])

    handle = None
    if layer_module is not None and hook_fn is not None:
        handle = layer_module.register_forward_hook(hook_fn)

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = out[0, prompt_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return text, int(generated.numel())

    finally:
        if handle is not None:
            handle.remove()


# ---------------------------------------------------------------------------
# Resume / output helpers
# ---------------------------------------------------------------------------

def existing_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    return {
        (
            str(row["prompt_id"]),
            str(row["condition"]),
            int(row.get("seed", PROBE_SEED)),
        )
        for row in read_jsonl(path)
    }


def save_generation(
    path: Path,
    example: dict,
    model_key: str,
    experiment: str,
    condition: str,
    response_text: str,
    generated_tokens: int,
    seed: int,
    layer_0based: int | None = None,
    alpha: float | None = None,
    lambda_value: float | None = None,
    direction_kind: str | None = None,
    projection_scale: float | None = None,
    probe_val_auroc: float | None = None,
    record: InterventionRecord | None = None,
):
    row = {
        "prompt_id": example["prompt_id"],
        "manifest_id": example.get("manifest_id"),
        "source_file": example.get("source_file"),
        "source_line": example.get("source_line"),
        "raw_index": example.get("raw_index"),
        "model": model_key,
        "condition": condition,
        "experiment": experiment,
        "evidence_label": example["label"],
        "response_text": response_text,
        "generated_tokens": generated_tokens,
        "seed": seed,
        "layer_0based": layer_0based,
        "layer_1based": None if layer_0based is None else layer_0based + 1,
        "alpha": alpha,
        "lambda": lambda_value,
        "direction_kind": direction_kind,
        "projection_scale": projection_scale,
        "probe_val_auroc": probe_val_auroc,
        "intervention_scope": "prefill_last_prompt_token",
    }
    if record is not None:
        row.update(asdict(record))
    append_jsonl(path, row)


def run_no_intervention_if_needed(
    example: dict,
    bundle: dict,
    args,
    output_file: Path,
    done: set,
    experiment_name: str,
):
    condition = "no_intervention"
    key = (example["prompt_id"], condition, args.seed)
    if key in done:
        return

    response, n_tok = generate_one(
        bundle=bundle,
        model_key=args.model_key,
        prompt=example["prompt"],
        layer_module=None,
        hook_fn=None,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    save_generation(
        output_file,
        example,
        args.model_key,
        experiment_name,
        condition,
        response,
        n_tok,
        args.seed,
    )
    done.add(key)


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------

def run_experiment_1(
    args,
    bundle,
    layers,
    examples,
    selected_layer,
    selected_cav,
    selected_sigma,
    layer_auc,
):
    output_file = args.output_dir / "expt1_generations.jsonl"
    done = existing_keys(output_file)
    random_dir = matched_random_direction(selected_cav, args.random_seed)

    print(
        f"Expt1 selected layer: 0-based={selected_layer}, "
        f"paper={selected_layer + 1}, projection_std={selected_sigma:.6g}"
    )

    for i, ex in enumerate(examples, 1):
        print(f"[Expt1 {i}/{len(examples)}] {ex['prompt_id']} label={ex['label']}")
        run_no_intervention_if_needed(
            ex, bundle, args, output_file, done, "direction"
        )

        for alpha in args.alphas:
            for kind, direction in (("cav", selected_cav), ("random", random_dir)):
                condition = (
                    f"expt1_{kind}_add_a{alpha:+g}_"
                    f"l{selected_layer}"
                )
                key = (ex["prompt_id"], condition, args.seed)
                if key in done:
                    continue

                record = InterventionRecord()
                magnitude = float(alpha) * float(selected_sigma)
                hook = make_prefill_last_token_hook(
                    mode="add",
                    intervention_direction=direction,
                    cav_direction=selected_cav,
                    magnitude=magnitude,
                    record=record,
                )

                response, n_tok = generate_one(
                    bundle,
                    args.model_key,
                    ex["prompt"],
                    layers[selected_layer],
                    hook,
                    args.max_new_tokens,
                    args.seed,
                )

                save_generation(
                    output_file,
                    ex,
                    args.model_key,
                    "direction",
                    condition,
                    response,
                    n_tok,
                    args.seed,
                    layer_0based=selected_layer,
                    alpha=alpha,
                    direction_kind=kind,
                    projection_scale=selected_sigma,
                    probe_val_auroc=layer_auc.get(selected_layer),
                    record=record,
                )
                done.add(key)


# ---------------------------------------------------------------------------
# Experiment 2 - local CAV at every layer
# ---------------------------------------------------------------------------

def parse_layer_spec(
    spec: str,
    n_layers: int,
    selected_layer: int,
) -> list[int]:
    spec = spec.strip().lower()

    if spec == "all":
        return list(range(n_layers))

    if spec == "selected":
        return [selected_layer]

    if spec == "window":
        candidates = [
            selected_layer - 4,
            selected_layer - 2,
            selected_layer,
            selected_layer + 2,
            selected_layer + 4,
        ]
        return sorted(
            set(
                max(0, min(n_layers - 1, x))
                for x in candidates
            )
        )

    values = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue

        if ":" in token:
            parts = [int(x) for x in token.split(":")]
            if len(parts) == 2:
                start, stop = parts
                step = 1
            elif len(parts) == 3:
                start, stop, step = parts
            else:
                raise ValueError(f"Invalid layer range: {token}")
            values.extend(range(start, stop, step))
        else:
            values.append(int(token))

    values = sorted(set(values))
    for layer in values:
        if not 0 <= layer < n_layers:
            raise ValueError(f"Layer {layer} outside [0,{n_layers - 1}]")
    return values


def run_experiment_2(
    args,
    bundle,
    layers,
    examples,
    selected_layer,
    layer_bank,
    layer_auc,
):
    output_file = args.output_dir / "expt2_layer_sweep.jsonl"
    done = existing_keys(output_file)
    sweep_layers = parse_layer_spec(
        args.layers,
        len(layers),
        selected_layer,
    )

    cavs = layer_bank["cavs"]
    sigmas = layer_bank["projection_stds"]

    print(f"Expt2 layers (0-based): {sweep_layers}")
    print(f"Expt2 paper layers:     {[x + 1 for x in sweep_layers]}")
    print("Expt2 uses LOCAL c_l from a separately trained probe at every layer.")

    for i, ex in enumerate(examples, 1):
        print(f"[Expt2 {i}/{len(examples)}] {ex['prompt_id']} label={ex['label']}")
        run_no_intervention_if_needed(
            ex, bundle, args, output_file, done, "layer_sweep"
        )

        for layer in sweep_layers:
            local_cav = torch.from_numpy(cavs[layer]).float()
            local_sigma = float(sigmas[layer])

            directions = [("cav", local_cav)]
            if args.include_random:
                directions.append(
                    (
                        "random",
                        matched_random_direction(
                            local_cav, args.random_seed + layer
                        ),
                    )
                )

            for alpha in args.alphas:
                for kind, direction in directions:
                    condition = (
                        f"expt2_{kind}_localcav_a{alpha:+g}_"
                        f"l{layer}"
                    )
                    key = (ex["prompt_id"], condition, args.seed)
                    if key in done:
                        continue

                    record = InterventionRecord()
                    magnitude = float(alpha) * local_sigma
                    hook = make_prefill_last_token_hook(
                        mode="add",
                        intervention_direction=direction,
                        cav_direction=local_cav,
                        magnitude=magnitude,
                        record=record,
                    )

                    response, n_tok = generate_one(
                        bundle,
                        args.model_key,
                        ex["prompt"],
                        layers[layer],
                        hook,
                        args.max_new_tokens,
                        args.seed,
                    )

                    save_generation(
                        output_file,
                        ex,
                        args.model_key,
                        "layer_sweep",
                        condition,
                        response,
                        n_tok,
                        args.seed,
                        layer_0based=layer,
                        alpha=alpha,
                        direction_kind=kind,
                        projection_scale=local_sigma,
                        probe_val_auroc=layer_auc.get(layer),
                        record=record,
                    )
                    done.add(key)


# ---------------------------------------------------------------------------
# Experiment 3
# ---------------------------------------------------------------------------

def run_experiment_3(
    args,
    bundle,
    layers,
    examples,
    selected_layer,
    selected_cav,
    train_mean,
    layer_auc,
):
    output_file = args.output_dir / "expt3_ablation.jsonl"
    done = existing_keys(output_file)
    random_dir = matched_random_direction(selected_cav, args.random_seed)

    print(
        f"Expt3 selected layer: 0-based={selected_layer}, "
        f"paper={selected_layer + 1}"
    )

    for i, ex in enumerate(examples, 1):
        print(f"[Expt3 {i}/{len(examples)}] {ex['prompt_id']} label={ex['label']}")
        run_no_intervention_if_needed(
            ex, bundle, args, output_file, done, "ablation"
        )

        for lam in args.lambdas:
            for kind, direction in (("cav", selected_cav), ("random", random_dir)):
                condition = (
                    f"expt3_{kind}_ablate_lam{lam:g}_"
                    f"l{selected_layer}"
                )
                key = (ex["prompt_id"], condition, args.seed)
                if key in done:
                    continue

                record = InterventionRecord()
                hook = make_prefill_last_token_hook(
                    mode="ablate",
                    intervention_direction=direction,
                    cav_direction=selected_cav,
                    magnitude=float(lam),
                    record=record,
                    train_mean=train_mean,
                )

                response, n_tok = generate_one(
                    bundle,
                    args.model_key,
                    ex["prompt"],
                    layers[selected_layer],
                    hook,
                    args.max_new_tokens,
                    args.seed,
                )

                save_generation(
                    output_file,
                    ex,
                    args.model_key,
                    "ablation",
                    condition,
                    response,
                    n_tok,
                    args.seed,
                    layer_0based=selected_layer,
                    lambda_value=lam,
                    direction_kind=kind,
                    probe_val_auroc=layer_auc.get(selected_layer),
                    record=record,
                )
                done.add(key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--experiment", required=True, choices=["1", "2", "3"])
    p.add_argument("--model-key", required=True)
    p.add_argument("--model-path", required=True)

    p.add_argument("--input-jsonl", required=True, type=Path)

    p.add_argument("--train-activation-dir", required=True, type=Path)
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Immutable HealthBench evidence-sufficiency manifest. Also used to recover the frozen validation split.",
    )

    p.add_argument(
        "--probe-root",
        type=Path,
        default=Path("outputs/probes"),
        help="Must contain summary.csv and <model>/cav.npy.",
    )

    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=160)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-seed", type=int, default=12345)

    p.add_argument(
        "--alphas",
        type=parse_float_list,
        default=parse_float_list("-2,-1,-0.5,0.5,1,2"),
    )

    p.add_argument(
        "--layers",
        default="window",
        help="Experiment 2 only: selected | window | all | comma list | start:stop:step.",
    )
    p.add_argument(
        "--include-random",
        action="store_true",
        help="Experiment 2: include a local matched-random control at every layer.",
    )
    p.add_argument(
        "--force-rebuild-layer-cavs",
        action="store_true",
        help="Experiment 2: rebuild outputs/probes/<model>/layerwise_cavs.npz.",
    )

    p.add_argument(
        "--lambdas",
        type=parse_float_list,
        default=parse_float_list("0.5,1.0"),
    )

    return p


def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BRIDGE CAUSAL INTERVENTIONS")
    print("=" * 80)
    print(f"Experiment: {args.experiment}")
    print(f"Model:      {args.model_key}")
    print(f"Model path: {args.model_path}")
    print(f"Output:     {args.output_dir}")
    print("=" * 80)

    # Canonical frozen probe artifacts.
    selected_layer, selected_layer_1based = load_selected_layer(
        args.probe_root, args.model_key
    )
    selected_cav = load_selected_cav(args.probe_root, args.model_key)
    layer_auc = load_layer_validation_auc(args.probe_root, args.model_key)

    print(
        f"Frozen selected layer from probe summary: "
        f"0-based={selected_layer}, paper layer={selected_layer_1based}"
    )

    examples = load_eval_examples(
        args.input_jsonl,
        args.manifest,
        args.max_per_class,
        args.seed,
    )
    if not examples:
        raise RuntimeError("No evaluation examples selected.")

    bundle = load_model_interface(args.model_path)
    layers = resolve_decoder_layers(bundle["model"], bundle["config"])

    expected_layers = infer_num_hidden_layers(bundle["config"])
    hidden_size = infer_hidden_size(bundle["config"])

    if expected_layers is not None and len(layers) != expected_layers:
        raise RuntimeError(
            f"Resolved {len(layers)} decoder layers, config says {expected_layers}."
        )

    if not 0 <= selected_layer < len(layers):
        raise RuntimeError(
            f"Selected layer {selected_layer} outside 0..{len(layers)-1}"
        )

    if hidden_size is not None and selected_cav.numel() != hidden_size:
        raise RuntimeError(
            f"Selected CAV dim={selected_cav.numel()} "
            f"but config hidden_size={hidden_size}."
        )

    n_layers = len(layers)
    hidden_dim = selected_cav.numel()

    metadata = {
        "experiment": args.experiment,
        "model_key": args.model_key,
        "model_path": args.model_path,
        "model_class": bundle["model"].__class__.__name__,
        "model_kind": bundle["kind"],
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "selected_layer_0based": selected_layer,
        "selected_layer_1based": selected_layer_1based,
        "selected_cav": str(args.probe_root / args.model_key / "cav.npy"),
        "train_activation_dir": str(args.train_activation_dir),
        "manifest": str(args.manifest) if args.manifest else None,
        "input_jsonl": str(args.input_jsonl),
        "n_examples": len(examples),
        "seed": args.seed,
        "random_seed": args.random_seed,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "intervention_scope": "prefill_last_prompt_token",
        "probe_C": PROBE_C,
        "probe_class_weight": "balanced",
        "probe_solver": "liblinear",
        "probe_max_iter": PROBE_MAX_ITER,
        "git_commit": git_commit(),
    }

    if args.experiment == "1":
        selected_sigma = compute_selected_projection_std(
            train_activation_dir=args.train_activation_dir,
            manifest=args.manifest,
            selected_cav=selected_cav,
            selected_layer=selected_layer,
            n_layers=n_layers,
        )
        metadata["alphas"] = args.alphas
        metadata["selected_projection_std"] = selected_sigma

        run_experiment_1(
            args,
            bundle,
            layers,
            examples,
            selected_layer,
            selected_cav,
            selected_sigma,
            layer_auc,
        )

    elif args.experiment == "2":
        layer_bank = build_or_load_layerwise_cavs(
            probe_root=args.probe_root,
            model_key=args.model_key,
            train_activation_dir=args.train_activation_dir,
            manifest=args.manifest,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            selected_layer=selected_layer,
            saved_selected_cav=selected_cav,
            force_rebuild=args.force_rebuild_layer_cavs,
        )

        metadata["layers"] = args.layers
        metadata["alphas"] = args.alphas
        metadata["include_random"] = args.include_random
        metadata["layerwise_cav_bank"] = str(
            args.probe_root / args.model_key / "layerwise_cavs.npz"
        )
        metadata["selected_cav_reconstruction_cosine"] = float(
            layer_bank["selected_cav_cosine"][0]
        )

        run_experiment_2(
            args,
            bundle,
            layers,
            examples,
            selected_layer,
            layer_bank,
            layer_auc,
        )

    else:
        train_mean = compute_train_mean_at_layer(
            train_activation_dir=args.train_activation_dir,
            manifest=args.manifest,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            layer=selected_layer,
        )
        metadata["lambdas"] = args.lambdas

        run_experiment_3(
            args,
            bundle,
            layers,
            examples,
            selected_layer,
            selected_cav,
            train_mean,
            layer_auc,
        )

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("CAUSAL EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Metadata: {args.output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
