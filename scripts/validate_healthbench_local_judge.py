#!/usr/bin/env python3
"""
Validate a local LLM-as-judge against HealthBench meta-evaluation physician labels.

This script mirrors the core HealthBench meta-eval protocol:
  * builds the grader prompt from the exact GRADER_TEMPLATE in the local
    openai/simple-evals checkout;
  * asks a fixed local Hugging Face judge for `criteria_met: true/false`;
  * compares each model prediction against every physician binary label;
  * reports the official-style balanced pairwise F1 as the primary metric;
  * additionally reports raw agreement, balanced accuracy, Cohen's kappa,
    majority-vote metrics, category breakdowns, physician-vs-peer reference,
    and row-clustered bootstrap 95% confidence intervals.

The run is resumable: predictions are appended to predictions.jsonl after each batch.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Utilities
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--meta-jsonl",
        type=Path,
        required=True,
        help="HealthBench 2025-05-07-06-14-12_oss_meta_eval.jsonl",
    )
    p.add_argument(
        "--simple-evals-repo",
        type=Path,
        required=True,
        help="Path to local openai/simple-evals checkout",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model id or local model path",
    )
    p.add_argument(
        "--judge-backend",
        type=str,
        choices=("auto", "causal_lm", "gemma3"),
        default="auto",
        help=(
            "Model-loading backend. 'auto' detects the architecture from AutoConfig; "
            "'causal_lm' forces AutoModelForCausalLM; 'gemma3' forces the official "
            "Gemma3ForConditionalGeneration + AutoProcessor path."
        ),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--max-input-tokens",
        type=int,
        default=0,
        help="0 = do not truncate. Nonzero values truncate from the left only as a last resort.",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="0 = all meta-eval rows; otherwise deterministic random sample.",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Matches simple-evals' Random(0) convention for subsampling.",
    )
    p.add_argument(
        "--exclude-indices-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON list of meta-eval row indices to exclude before "
            "subsampling. Useful for making a final validation subset disjoint "
            "from an earlier model-selection subset."
        ),
    )
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loading if needed.",
    )
    p.add_argument(
        "--limit-for-smoke",
        type=int,
        default=0,
        help="Optional final cap after sampling; intended only for quick smoke tests.",
    )
    return p.parse_args()


def find_healthbench_eval_py(repo: Path) -> Path:
    candidates = [
        repo / "healthbench_eval.py",
        repo / "simple_evals" / "healthbench_eval.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find healthbench_eval.py under {repo}. Tried: {candidates}"
    )


def extract_python_string_constant(pyfile: Path, variable_name: str) -> str:
    """
    Safely extract a top-level string assignment without importing the module.

    Supports the forms used by simple-evals, including:
      NAME = "..."
      NAME = "...".strip()
      NAME = "...".lstrip()
      NAME = "...".rstrip()
      NAME = "a" + "b"

    This intentionally supports only a very small AST subset so that reading the
    template cannot execute arbitrary code from the checked-out repository.
    """
    tree = ast.parse(pyfile.read_text(encoding="utf-8"), filename=str(pyfile))

    def eval_string_expr(node: ast.AST) -> str:
        # Plain string literal.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value

        # Compatibility with older Python ASTs.
        if isinstance(node, ast.Str):  # pragma: no cover
            return node.s

        # Concatenated literals, e.g. "a" + "b".
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return eval_string_expr(node.left) + eval_string_expr(node.right)

        # Whitelisted zero-argument string methods, e.g. "...".strip().
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and len(node.args) == 0
                and len(node.keywords) == 0
            ):
                base = eval_string_expr(node.func.value)
                method = node.func.attr
                if method == "strip":
                    return base.strip()
                if method == "lstrip":
                    return base.lstrip()
                if method == "rstrip":
                    return base.rstrip()

            raise ValueError(
                f"Unsupported function call while extracting {variable_name}: "
                f"{ast.dump(node)}"
            )

        raise ValueError(
            f"Unsupported AST expression while extracting {variable_name}: "
            f"{ast.dump(node)}"
        )

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    value = eval_string_expr(node.value)
                    if not isinstance(value, str):
                        raise TypeError(f"{variable_name} is not a string in {pyfile}")
                    return value

        if isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == variable_name:
                if node.value is None:
                    raise ValueError(f"{variable_name} has no assigned value in {pyfile}")
                value = eval_string_expr(node.value)
                if not isinstance(value, str):
                    raise TypeError(f"{variable_name} is not a string in {pyfile}")
                return value

    raise KeyError(f"Could not locate {variable_name} in {pyfile}")

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def validate_meta_schema(rows: list[dict[str, Any]]) -> None:
    required = {
        "prompt",
        "completion",
        "rubric",
        "binary_labels",
        "anonymized_physician_ids",
        "category",
    }
    if not rows:
        raise ValueError("Meta-eval file is empty")
    missing = required - set(rows[0].keys())
    if missing:
        raise KeyError(f"Meta-eval rows are missing required fields: {sorted(missing)}")
    for i, row in enumerate(rows[:100]):
        labels = row["binary_labels"]
        ids = row["anonymized_physician_ids"]
        if not isinstance(labels, list) or len(labels) < 2:
            raise ValueError(f"Row {i}: binary_labels must contain >=2 physician labels")
        if len(labels) != len(ids):
            raise ValueError(f"Row {i}: label/id length mismatch")
        if any(type(x) is not bool for x in labels):
            raise ValueError(f"Row {i}: binary_labels must be booleans")


def load_excluded_indices(path: Path | None, n: int) -> set[int]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Excluded-indices JSON not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("--exclude-indices-json must contain a JSON list")

    excluded: set[int] = set()
    for x in raw:
        if isinstance(x, bool) or not isinstance(x, int):
            raise TypeError(
                "--exclude-indices-json must contain integer row indices only"
            )
        if not 0 <= x < n:
            raise ValueError(
                f"Excluded row index {x} is outside valid range [0, {n - 1}]"
            )
        excluded.add(x)

    return excluded


def select_indices(
    n: int,
    max_examples: int,
    seed: int,
    smoke_limit: int,
    excluded: set[int] | None = None,
) -> list[int]:
    excluded = excluded or set()
    idx = [i for i in range(n) if i not in excluded]

    if max_examples and max_examples < len(idx):
        rng = random.Random(seed)
        idx = rng.sample(idx, max_examples)

    if smoke_limit and smoke_limit < len(idx):
        idx = idx[:smoke_limit]

    return idx


def build_grader_prompt(row: dict[str, Any], template: str) -> str:
    convo = list(row["prompt"]) + [
        {"content": row["completion"], "role": "assistant"}
    ]
    convo_str = "\n\n".join(f"{m['role']}: {m['content']}" for m in convo)
    return template.replace("<<conversation>>", convo_str).replace(
        "<<rubric_item>>", str(row["rubric"])
    )


def parse_grader_output(text: str) -> tuple[bool | None, str | None]:
    """Leniently recover criteria_met and explanation from a HealthBench grader output."""
    stripped = text.strip()
    candidates = [stripped]

    # fenced JSON
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S | re.I)
    candidates.extend(fenced)

    # broad outer object fallback
    first = stripped.find("{")
    last = stripped.rfind("}")
    if 0 <= first < last:
        candidates.append(stripped[first : last + 1])

    for c in candidates:
        try:
            obj = json.loads(c)
            val = obj.get("criteria_met")
            if type(val) is bool:
                explanation = obj.get("explanation")
                return val, None if explanation is None else str(explanation)
        except Exception:
            pass

    # final regex fallback
    m = re.search(r'["\']?criteria_met["\']?\s*:\s*(true|false)', stripped, flags=re.I)
    if m:
        return m.group(1).lower() == "true", None
    return None, None


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


@dataclass
class Confusion:
    tp: float
    tn: float
    fp: float
    fn: float

    @property
    def n(self) -> float:
        return self.tp + self.tn + self.fp + self.fn


def metrics_from_confusion(c: Confusion) -> dict[str, float]:
    precision_pos = safe_div(c.tp, c.tp + c.fp)
    recall_pos = safe_div(c.tp, c.tp + c.fn)
    precision_neg = safe_div(c.tn, c.tn + c.fn)
    recall_neg = safe_div(c.tn, c.tn + c.fp)

    def f1(p: float, r: float) -> float:
        if math.isnan(p) or math.isnan(r):
            return float("nan")
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    f1_pos = f1(precision_pos, recall_pos)
    f1_neg = f1(precision_neg, recall_neg)
    balanced_f1 = (
        (f1_pos + f1_neg) / 2
        if not (math.isnan(f1_pos) or math.isnan(f1_neg))
        else float("nan")
    )
    accuracy = safe_div(c.tp + c.tn, c.n)
    balanced_accuracy = (
        (recall_pos + recall_neg) / 2
        if not (math.isnan(recall_pos) or math.isnan(recall_neg))
        else float("nan")
    )

    if c.n:
        p_pred_pos = (c.tp + c.fp) / c.n
        p_true_pos = (c.tp + c.fn) / c.n
        p_expected = p_pred_pos * p_true_pos + (1 - p_pred_pos) * (1 - p_true_pos)
        kappa = (accuracy - p_expected) / (1 - p_expected) if p_expected < 1 else float("nan")
    else:
        kappa = float("nan")

    return {
        "n_pairs": c.n,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "precision_neg": precision_neg,
        "recall_neg": recall_neg,
        "f1_neg": f1_neg,
        "balanced_f1": balanced_f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "cohen_kappa": kappa,
    }


def row_pairwise_confusion(pred: bool, physician_labels: list[bool]) -> np.ndarray:
    pos = int(sum(physician_labels))
    neg = len(physician_labels) - pos
    if pred:
        # tp, tn, fp, fn
        return np.array([pos, 0, neg, 0], dtype=np.int64)
    return np.array([0, neg, 0, pos], dtype=np.int64)


def confusion_from_vector(v: np.ndarray) -> Confusion:
    return Confusion(tp=float(v[0]), tn=float(v[1]), fp=float(v[2]), fn=float(v[3]))


def majority_label(labels: list[bool]) -> bool | None:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == n_neg:
        return None
    return n_pos > n_neg


def bootstrap_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


def clustered_bootstrap(
    pairwise_rows: np.ndarray,
    majority_rows: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, tuple[float, float]], list[dict[str, float]]]:
    """Bootstrap meta-eval ROWS, preserving within-row multiple physician labels."""
    rng = np.random.default_rng(seed)
    n = len(pairwise_rows)
    records: list[dict[str, float]] = []
    if n_boot <= 0:
        return {}, records

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pair_sum = pairwise_rows[idx].sum(axis=0)
        pair_m = metrics_from_confusion(confusion_from_vector(pair_sum))

        maj_sample = majority_rows[idx]
        valid = maj_sample[:, 4] == 1
        if valid.any():
            maj_sum = maj_sample[valid, :4].sum(axis=0)
            maj_m = metrics_from_confusion(confusion_from_vector(maj_sum))
        else:
            maj_m = {"accuracy": float("nan"), "balanced_accuracy": float("nan"), "balanced_f1": float("nan"), "cohen_kappa": float("nan")}

        records.append(
            {
                "bootstrap_id": b,
                "pairwise_balanced_f1": pair_m["balanced_f1"],
                "pairwise_accuracy": pair_m["accuracy"],
                "pairwise_balanced_accuracy": pair_m["balanced_accuracy"],
                "pairwise_cohen_kappa": pair_m["cohen_kappa"],
                "majority_balanced_f1": maj_m["balanced_f1"],
                "majority_accuracy": maj_m["accuracy"],
                "majority_balanced_accuracy": maj_m["balanced_accuracy"],
                "majority_cohen_kappa": maj_m["cohen_kappa"],
            }
        )

    ci: dict[str, tuple[float, float]] = {}
    for key in records[0]:
        if key == "bootstrap_id":
            continue
        ci[key] = bootstrap_ci([r[key] for r in records])
    return ci, records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_cached_predictions(path: Path) -> dict[int, dict[str, Any]]:
    cache: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("model_predicted_positive") is not None:
                cache[int(obj["row_index"])] = obj
    return cache


def model_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


@dataclass
class JudgeRuntime:
    """Loaded judge plus the model-specific prompt renderer/tokenizer."""

    model: Any
    tokenizer: Any
    prompt_renderer: Any
    model_type: str
    backend: str


def inspect_judge_architecture(
    model_name: str,
    trust_remote_code: bool,
) -> tuple[Any, str]:
    """
    Load only the Hugging Face config so we can choose the correct runtime.

    Most text-only instruction models (Qwen, Llama, Mistral, etc.) use the
    generic AutoModelForCausalLM path. Gemma 3 4B/12B/27B checkpoints are
    composite multimodal checkpoints, so their official loader is
    Gemma3ForConditionalGeneration + AutoProcessor even when this evaluation
    supplies text only.
    """
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    model_type = str(getattr(config, "model_type", "") or "")
    return config, model_type


def resolve_judge_backend(model_type: str, requested_backend: str) -> str:
    if requested_backend != "auto":
        return requested_backend

    if model_type == "gemma3":
        return "gemma3"

    return "causal_lm"


def load_judge_runtime(
    model_name: str,
    dtype: torch.dtype,
    trust_remote_code: bool,
    requested_backend: str,
) -> JudgeRuntime:
    """
    Architecture-aware judge loader.

    Supported paths:
      * causal_lm:
          Generic text-only Hugging Face causal LM path. This covers Qwen,
          Llama, Mistral, Gemma/Gemma2, and most other decoder-only models.
      * gemma3:
          Official composite Gemma 3 path using
          Gemma3ForConditionalGeneration + AutoProcessor. We still feed only
          text; loading the checkpoint this way avoids relying on brittle
          text-config extraction behavior across Transformers versions.

    Add future architecture-specific adapters here only when the generic
    AutoModelForCausalLM path is insufficient.
    """
    _, model_type = inspect_judge_architecture(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    backend = resolve_judge_backend(model_type, requested_backend)

    print(
        f"[INFO] Judge architecture: model_type={model_type!r}, "
        f"backend={backend!r}"
    )

    common_model_kwargs = {
        "dtype": dtype,
        "device_map": {"": 0},
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    if backend == "gemma3":
        if model_type != "gemma3":
            raise ValueError(
                f"--judge-backend gemma3 was requested, but model_type={model_type!r}"
            )

        try:
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        except ImportError as e:
            raise RuntimeError(
                "Gemma 3 requires a Transformers version with "
                "Gemma3ForConditionalGeneration support. Upgrade transformers "
                "(Gemma 3 support starts in the 4.50 series; use a current "
                "stable release for the bug fixes around composite configs)."
            ) from e

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "Gemma 3 AutoProcessor did not expose a tokenizer; "
                "cannot perform text-only grading."
            )

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            **common_model_kwargs,
        )
        prompt_renderer = processor

    elif backend == "causal_lm":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **common_model_kwargs,
        )
        prompt_renderer = tokenizer

    else:
        raise ValueError(f"Unsupported judge backend: {backend}")

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "Tokenizer has neither pad_token_id nor eos_token_id; "
                "cannot configure batched generation safely."
            )
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    model.eval()

    return JudgeRuntime(
        model=model,
        tokenizer=tokenizer,
        prompt_renderer=prompt_renderer,
        model_type=model_type,
        backend=backend,
    )


def render_judge_prompt(runtime: JudgeRuntime, prompt: str) -> str:
    """
    Render one HealthBench grader prompt using the model's native chat template.

    Gemma 3's processor expects multimodal-style content blocks, even though
    this evaluator supplies text only. Generic text LMs use ordinary
    role/content strings.
    """
    if runtime.backend == "gemma3":
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        return runtime.prompt_renderer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    messages = [{"role": "user", "content": prompt}]
    chat_template = getattr(runtime.tokenizer, "chat_template", None)

    if chat_template:
        return runtime.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Conservative fallback for an unusual tokenizer without a chat template.
    return "USER:\n" + prompt + "\n\nASSISTANT:\n"


def model_input_device(model: Any) -> torch.device:
    """
    Return the device on which input IDs should be placed.

    The validation jobs use one GPU and device_map={'': 0}, but deriving the
    device from the input embedding is more robust than relying on model.device
    for every architecture.
    """
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def generate_texts(
    runtime: JudgeRuntime,
    prompts: list[str],
    max_new_tokens: int,
    max_input_tokens: int,
) -> list[str]:
    rendered = [render_judge_prompt(runtime, p) for p in prompts]

    tok_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        # Chat templates already insert the model's required BOS/EOS/control
        # tokens. Re-adding tokenizer special tokens here can duplicate them.
        "add_special_tokens": False,
    }
    if max_input_tokens > 0:
        # Left truncation preserves the rubric + end of the conversation, but any
        # truncation should be disclosed. Default is 0 (no truncation).
        runtime.tokenizer.truncation_side = "left"
        tok_kwargs.update(
            {
                "truncation": True,
                "max_length": max_input_tokens,
            }
        )

    batch = runtime.tokenizer(rendered, **tok_kwargs)
    device = model_input_device(runtime.model)
    batch = {k: v.to(device) for k, v in batch.items()}
    input_width = batch["input_ids"].shape[1]

    generation_kwargs: dict[str, Any] = {
        **batch,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": runtime.tokenizer.pad_token_id,
    }

    # Some tokenizers expose multiple EOS IDs through the model's generation
    # config. Passing None explicitly is less robust than simply omitting it.
    if runtime.tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = runtime.tokenizer.eos_token_id

    with torch.inference_mode():
        out = runtime.model.generate(**generation_kwargs)

    return [
        runtime.tokenizer.decode(
            seq[input_width:],
            skip_special_tokens=True,
        ).strip()
        for seq in out
    ]


def generate_with_oom_split(
    runtime: JudgeRuntime,
    prompts: list[str],
    max_new_tokens: int,
    max_input_tokens: int,
) -> list[str]:
    """
    Generate a batch; recursively split it if a batch-level CUDA OOM occurs.

    A single-example OOM is re-raised because the model/input itself does not
    fit and silently retrying cannot fix that.
    """
    try:
        return generate_texts(
            runtime,
            prompts,
            max_new_tokens,
            max_input_tokens,
        )
    except torch.OutOfMemoryError:
        if len(prompts) == 1:
            raise

        torch.cuda.empty_cache()
        mid = len(prompts) // 2

        left = generate_with_oom_split(
            runtime,
            prompts[:mid],
            max_new_tokens,
            max_input_tokens,
        )
        right = generate_with_oom_split(
            runtime,
            prompts[mid:],
            max_new_tokens,
            max_input_tokens,
        )
        return left + right


def summarize_physician_peer_reference(
    selected_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Official-style physician-vs-other-physicians pairwise balanced F1, per physician."""
    per_physician: dict[str, list[np.ndarray]] = defaultdict(list)

    for row in selected_rows:
        labels = row["binary_labels"]
        ids = row["anonymized_physician_ids"]
        for i, (phys_id, self_label) in enumerate(zip(ids, labels, strict=True)):
            others = labels[:i] + labels[i + 1 :]
            if others:
                per_physician[str(phys_id)].append(row_pairwise_confusion(self_label, others))

    rows: list[dict[str, Any]] = []
    balanced_f1_values: list[float] = []
    for phys_id, vectors in per_physician.items():
        total = np.vstack(vectors).sum(axis=0)
        m = metrics_from_confusion(confusion_from_vector(total))
        rows.append({"physician_id": phys_id, **m})
        if np.isfinite(m["balanced_f1"]):
            balanced_f1_values.append(m["balanced_f1"])

    summary = {
        "n_physicians": float(len(rows)),
        "physician_peer_balanced_f1_mean": float(np.mean(balanced_f1_values)) if balanced_f1_values else float("nan"),
        "physician_peer_balanced_f1_sd": float(np.std(balanced_f1_values, ddof=1)) if len(balanced_f1_values) > 1 else float("nan"),
        "physician_peer_balanced_f1_median": float(np.median(balanced_f1_values)) if balanced_f1_values else float("nan"),
    }
    rows.sort(key=lambda x: str(x["physician_id"]))
    return rows, summary


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.meta_jsonl.exists():
        raise FileNotFoundError(args.meta_jsonl)
    hb_eval_py = find_healthbench_eval_py(args.simple_evals_repo)
    grader_template = extract_python_string_constant(hb_eval_py, "GRADER_TEMPLATE")

    required_placeholders = ("<<conversation>>", "<<rubric_item>>")
    missing_placeholders = [
        placeholder for placeholder in required_placeholders
        if placeholder not in grader_template
    ]
    if missing_placeholders:
        raise ValueError(
            "Extracted GRADER_TEMPLATE is missing expected placeholder(s): "
            f"{missing_placeholders}. Source: {hb_eval_py}"
        )

    print(f"[INFO] Meta-eval: {args.meta_jsonl}")
    print(f"[INFO] simple-evals HealthBench source: {hb_eval_py}")
    print(f"[INFO] GRADER_TEMPLATE sha256: {sha256_text(grader_template)}")

    rows = load_jsonl(args.meta_jsonl)
    validate_meta_schema(rows)

    excluded_indices = load_excluded_indices(
        args.exclude_indices_json,
        n=len(rows),
    )
    selected_indices = select_indices(
        len(rows),
        args.max_examples,
        args.sample_seed,
        args.limit_for_smoke,
        excluded=excluded_indices,
    )
    selected_rows = [rows[i] for i in selected_indices]
    print(
        f"[INFO] Loaded {len(rows)} total rows; "
        f"excluded {len(excluded_indices)}; "
        f"selected {len(selected_rows)}"
    )

    _, judge_model_type = inspect_judge_architecture(
        args.judge_model,
        trust_remote_code=args.trust_remote_code,
    )
    judge_backend = resolve_judge_backend(
        judge_model_type,
        args.judge_backend,
    )
    print(
        f"[INFO] Judge config: model_type={judge_model_type!r}, "
        f"backend={judge_backend!r}"
    )

    selected_hash = sha256_text(",".join(map(str, selected_indices)))
    config_path = args.output_dir / "run_config.json"
    current_config = {
        "meta_jsonl": str(args.meta_jsonl.resolve()),
        "meta_jsonl_sha256": sha256_file(args.meta_jsonl),
        "simple_evals_healthbench_eval_py": str(hb_eval_py.resolve()),
        "grader_template_sha256": sha256_text(grader_template),
        "judge_model": args.judge_model,
        "judge_model_type": judge_model_type,
        "judge_backend": judge_backend,
        "max_examples": args.max_examples,
        "sample_seed": args.sample_seed,
        "exclude_indices_json": (
            str(args.exclude_indices_json.resolve())
            if args.exclude_indices_json is not None
            else None
        ),
        "excluded_indices_sha256": (
            sha256_text(",".join(map(str, sorted(excluded_indices))))
            if excluded_indices
            else None
        ),
        "n_excluded": len(excluded_indices),
        "smoke_limit": args.limit_for_smoke,
        "selected_indices_sha256": selected_hash,
        "n_selected": len(selected_indices),
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
    }
    if config_path.exists():
        old = json.loads(config_path.read_text())
        compare_keys = [
            "meta_jsonl_sha256",
            "grader_template_sha256",
            "judge_model",
            "judge_model_type",
            "judge_backend",
            "excluded_indices_sha256",
            "selected_indices_sha256",
            "max_new_tokens",
            "max_input_tokens",
        ]
        mismatch = [k for k in compare_keys if old.get(k) != current_config.get(k)]
        if mismatch:
            raise RuntimeError(
                "Existing output-dir belongs to a different run. "
                f"Mismatched keys: {mismatch}. Use a new --output-dir or remove the old one."
            )
    else:
        config_path.write_text(json.dumps(current_config, indent=2), encoding="utf-8")
        (args.output_dir / "selected_indices.json").write_text(
            json.dumps(selected_indices), encoding="utf-8"
        )

    pred_path = args.output_dir / "predictions.jsonl"
    cached = load_cached_predictions(pred_path)
    remaining = [i for i in selected_indices if i not in cached]
    print(f"[INFO] Cached predictions: {len(cached)}; remaining: {len(remaining)}")

    if remaining:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for the configured local judge run")

        dtype = model_dtype()
        print(f"[INFO] Loading judge {args.judge_model} with dtype={dtype}")

        runtime = load_judge_runtime(
            model_name=args.judge_model,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            requested_backend=args.judge_backend,
        )

        t0 = time.time()
        with pred_path.open("a", encoding="utf-8") as fout:
            for start in range(0, len(remaining), args.batch_size):
                batch_idx = remaining[start : start + args.batch_size]
                prompts = [build_grader_prompt(rows[i], grader_template) for i in batch_idx]
                texts = generate_with_oom_split(
                    runtime,
                    prompts,
                    args.max_new_tokens,
                    args.max_input_tokens,
                )

                for row_idx, raw_text in zip(batch_idx, texts, strict=True):
                    pred, explanation = parse_grader_output(raw_text)
                    if pred is None:
                        raise ValueError(
                            f"Could not parse criteria_met for row {row_idx}. Raw output:\n{raw_text}"
                        )
                    row = rows[row_idx]
                    labels = row["binary_labels"]
                    obj = {
                        "row_index": row_idx,
                        "completion_id": row.get("completion_id"),
                        "category": row["category"],
                        "model_predicted_positive": pred,
                        "num_physician_labels": len(labels),
                        "percent_physician_positive": float(sum(labels) / len(labels)),
                        "physician_majority_label": majority_label(labels),
                        "physician_labels": labels,
                        "anonymized_physician_ids": row["anonymized_physician_ids"],
                        "explanation": explanation,
                        "raw_grader_output": raw_text,
                    }
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    fout.flush()
                    cached[row_idx] = obj

                done = start + len(batch_idx)
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0.0
                print(
                    f"[INFO] {done}/{len(remaining)} newly graded "
                    f"({rate:.2f} rows/s); total cached={len(cached)}",
                    flush=True,
                )

        del runtime.model
        del runtime
        torch.cuda.empty_cache()

    # Re-load cache from disk and make sure every selected row is present.
    cached = load_cached_predictions(pred_path)
    missing = [i for i in selected_indices if i not in cached]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} predictions after grading; first: {missing[:10]}")

    pairwise_row_vectors: list[np.ndarray] = []
    majority_row_vectors: list[np.ndarray] = []
    category_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    category_item_counts: dict[str, int] = defaultdict(int)
    category_judgment_counts: dict[str, int] = defaultdict(int)

    pred_positive_count = 0
    n_ties = 0

    for row_idx in selected_indices:
        row = rows[row_idx]
        pred = bool(cached[row_idx]["model_predicted_positive"])
        labels = row["binary_labels"]
        category = str(row["category"])
        pred_positive_count += int(pred)

        pv = row_pairwise_confusion(pred, labels)
        pairwise_row_vectors.append(pv)
        category_vectors[category].append(pv)
        category_item_counts[category] += 1
        category_judgment_counts[category] += len(labels)

        maj = majority_label(labels)
        if maj is None:
            n_ties += 1
            majority_row_vectors.append(np.array([0, 0, 0, 0, 0], dtype=np.int64))
        else:
            mv = row_pairwise_confusion(pred, [maj])
            majority_row_vectors.append(np.concatenate([mv, np.array([1], dtype=np.int64)]))

    pair_arr = np.vstack(pairwise_row_vectors)
    maj_arr = np.vstack(majority_row_vectors)

    pair_metrics = metrics_from_confusion(confusion_from_vector(pair_arr.sum(axis=0)))
    maj_valid = maj_arr[:, 4] == 1
    majority_metrics = metrics_from_confusion(
        confusion_from_vector(maj_arr[maj_valid, :4].sum(axis=0))
    )

    ci, bootstrap_records = clustered_bootstrap(
        pair_arr, maj_arr, args.bootstrap, args.bootstrap_seed
    )
    write_csv(args.output_dir / "bootstrap_samples.csv", bootstrap_records)

    category_rows: list[dict[str, Any]] = []
    for category in sorted(category_vectors):
        v = np.vstack(category_vectors[category]).sum(axis=0)
        m = metrics_from_confusion(confusion_from_vector(v))
        category_rows.append(
            {
                "category": category,
                "n_items": category_item_counts[category],
                "n_physician_judgments": category_judgment_counts[category],
                **m,
            }
        )
    write_csv(args.output_dir / "category_metrics.csv", category_rows)

    physician_rows, physician_summary = summarize_physician_peer_reference(selected_rows)
    write_csv(args.output_dir / "physician_peer_reference.csv", physician_rows)

    n_phys_judgments = int(pair_arr.sum())
    all_phys_positive = int(pair_arr[:, 0].sum() + pair_arr[:, 3].sum())

    summary: dict[str, Any] = {
        "judge_model": args.judge_model,
        "n_meta_items": len(selected_indices),
        "n_physician_judgments": n_phys_judgments,
        "n_majority_ties_excluded": n_ties,
        "model_positive_rate_items": pred_positive_count / len(selected_indices),
        "physician_positive_rate_judgments": all_phys_positive / n_phys_judgments,
        # PRIMARY: official HealthBench meta-eval style pairwise balanced F1.
        "pairwise_precision_pos": pair_metrics["precision_pos"],
        "pairwise_recall_pos": pair_metrics["recall_pos"],
        "pairwise_f1_pos": pair_metrics["f1_pos"],
        "pairwise_precision_neg": pair_metrics["precision_neg"],
        "pairwise_recall_neg": pair_metrics["recall_neg"],
        "pairwise_f1_neg": pair_metrics["f1_neg"],
        "pairwise_balanced_f1": pair_metrics["balanced_f1"],
        "pairwise_accuracy": pair_metrics["accuracy"],
        "pairwise_balanced_accuracy": pair_metrics["balanced_accuracy"],
        "pairwise_cohen_kappa": pair_metrics["cohen_kappa"],
        # Secondary: strict physician-majority labels; ties excluded.
        "majority_n_items": int(maj_valid.sum()),
        "majority_accuracy": majority_metrics["accuracy"],
        "majority_balanced_accuracy": majority_metrics["balanced_accuracy"],
        "majority_f1_pos": majority_metrics["f1_pos"],
        "majority_f1_neg": majority_metrics["f1_neg"],
        "majority_balanced_f1": majority_metrics["balanced_f1"],
        "majority_cohen_kappa": majority_metrics["cohen_kappa"],
        **physician_summary,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
    }

    # Add clustered bootstrap CIs.
    ci_key_map = {
        "pairwise_balanced_f1": "pairwise_balanced_f1",
        "pairwise_accuracy": "pairwise_accuracy",
        "pairwise_balanced_accuracy": "pairwise_balanced_accuracy",
        "pairwise_cohen_kappa": "pairwise_cohen_kappa",
        "majority_balanced_f1": "majority_balanced_f1",
        "majority_accuracy": "majority_accuracy",
        "majority_balanced_accuracy": "majority_balanced_accuracy",
        "majority_cohen_kappa": "majority_cohen_kappa",
    }
    for summary_key, boot_key in ci_key_map.items():
        if boot_key in ci:
            lo, hi = ci[boot_key]
            summary[f"{summary_key}_ci95_low"] = lo
            summary[f"{summary_key}_ci95_high"] = hi

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    write_csv(args.output_dir / "summary.csv", [summary])

    metadata = {
        **current_config,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "architecture_policy": (
            "AutoModelForCausalLM for generic causal LMs; "
            "Gemma3ForConditionalGeneration + AutoProcessor for model_type='gemma3'."
        ),
        "primary_metric": "pairwise_balanced_f1",
        "primary_metric_definition": (
            "Macro-average of positive-class and negative-class F1 after pairing the fixed "
            "judge prediction for each meta-eval item with every physician binary label on that item."
        ),
        "bootstrap_unit": "meta-eval row (clustered over all physician labels within row)",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n========== HealthBench local-judge validation ==========")
    print(f"Meta items:              {summary['n_meta_items']}")
    print(f"Physician judgments:     {summary['n_physician_judgments']}")
    print(f"Pairwise balanced F1:    {summary['pairwise_balanced_f1']:.4f}")
    print(
        "  95% row-bootstrap CI: "
        f"[{summary.get('pairwise_balanced_f1_ci95_low', float('nan')):.4f}, "
        f"{summary.get('pairwise_balanced_f1_ci95_high', float('nan')):.4f}]"
    )
    print(f"Pairwise raw agreement:  {summary['pairwise_accuracy']:.4f}")
    print(f"Pairwise Cohen kappa:    {summary['pairwise_cohen_kappa']:.4f}")
    print(f"Majority balanced F1:    {summary['majority_balanced_f1']:.4f}")
    print(f"Majority Cohen kappa:    {summary['majority_cohen_kappa']:.4f}")
    print(
        "Physician peer F1:      "
        f"mean={summary['physician_peer_balanced_f1_mean']:.4f}, "
        f"median={summary['physician_peer_balanced_f1_median']:.4f}, "
        f"n={int(summary['n_physicians'])}"
    )
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
