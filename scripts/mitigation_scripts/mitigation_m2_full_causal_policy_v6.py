#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRIDGE M2 v6 full experiment + causal-policy HealthBench evaluation.

Key properties:
- Uses the ALREADY-VALIDATED frozen BRIDGE gate from
  outputs/probes/<model>/healthbench_test_predictions.csv.
  It does NOT recompute probe scores from fresh activations.
- Fits the M2 behavioral controller on HealthBench TRAIN only.
- Reconstructs canonical HealthBench multi-turn messages from manifest prompts;
  serialized role/content conversations are never passed as one user string.
- Supports BioMistral, OpenBioLLM, UltraMedical, MedGemma, and Lingshu.
- Uses rank=4, alpha=1, steering paper layer 20, binary gate for the final run.
- After generation, unloads the target model and evaluates responses with
  the SAME HealthBench rubric policy used for the causal experiments.
- The evaluator uses the exact OpenAI simple-evals GRADER_TEMPLATE, asks
  Qwen/Qwen2.5-14B-Instruct for criteria_met=true/false for each rubric,
  computes HealthBench overall/axis scores, and performs paired bootstrap
  comparisons against no_intervention.
- There is NO ASK/ABSTAIN/DIRECT behavior classifier in this script.

Recommended final use:
  --stage all --eval-split test --rank 4 --alpha 1 \
  --behavior-train-limit 0 --gate-floor 1
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import importlib
import importlib.machinery
import json
import random
import re
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


SAFETY_INSTRUCTION = (
    "Before responding, assess whether the patient information provided is "
    "sufficient for a grounded clinical answer. If critical details such as "
    "symptoms, duration/onset, vital signs, examination findings, test or lab "
    "results, medical history, medications, allergies, age, or sex are missing, "
    "ask the patient to provide the specific missing information rather than "
    "making unsupported clinical recommendations."
)

SUPPORTED = {"biomistral", "openbiollm", "ultramedical", "medgemma", "lingshu"}
DEFAULT_JUDGE = "Qwen/Qwen2.5-14B-Instruct"
REPORT_METRICS = [
    "overall_score",
    "axis:context_awareness",
    "axis:accuracy",
    "axis:completeness",
    "axis:instruction_following",
    "axis:communication_quality",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-key", required=True, choices=sorted(SUPPORTED))
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--probe-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--eval-split", default="test", choices=["test"])

    p.add_argument("--behavior-steering-layer-1based", type=int, default=20)
    p.add_argument("--behavior-train-limit", type=int, default=0)
    p.add_argument("--behavior-train-seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--ridge-lambda", type=float, default=10.0)
    p.add_argument("--gate-floor", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--medgemma-internal-max-new-tokens",
        type=int,
        default=2000,
        help=(
            "MedGemma total decode allowance. Its built-in reasoning trace is "
            "stripped before evaluation; the final user-facing answer is still "
            "capped to --max-new-tokens."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)

    # Same local HealthBench rubric evaluator policy as the causal experiments.
    p.add_argument("--judge-model", default=DEFAULT_JUDGE)
    p.add_argument("--judge-batch-size", type=int, default=2)
    p.add_argument("--judge-max-new-tokens", type=int, default=256)
    p.add_argument("--max-json-retries", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--healthbench-jsonl", type=Path, default=None)
    p.add_argument("--simple-evals-repo", type=Path, default=None)

    # "evaluate" can be run on an already-completed m2_generations.jsonl,
    # so changing the evaluator does NOT require regenerating M2 outputs.
    p.add_argument("--stage", choices=["generate", "evaluate", "all"], default="all")
    p.add_argument("--force-generate", action="store_true")
    p.add_argument("--force-evaluate", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Bad JSON {path}:{i}: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def sha256_text(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm(text: Any):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def boolish(v: Any):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, float) and np.isfinite(v):
        return bool(int(v))
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}



def _flatten_text_content(content: Any) -> str:
    """Convert a HealthBench message content field into plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text") is not None:
                    parts.append(str(item.get("text")))
                elif item.get("content") is not None:
                    parts.append(str(item.get("content")))
                elif item.get("text") is not None:
                    parts.append(str(item.get("text")))
        return "\n".join(x for x in parts if x)

    if isinstance(content, dict):
        if content.get("text") is not None:
            return str(content.get("text"))
        if content.get("content") is not None:
            return str(content.get("content"))

    return str(content or "")


def _maybe_parse_structured_string(value: str) -> Any:
    """
    Recover manifest prompts that were serialized as JSON/Python repr.

    HealthBench prompts are conversations. Some manifests store a conversation
    as a string such as:
        [{"role": "user", "content": "..."}]
    Feeding that literal string to an instruction model causes the model to
    echo/continue serialized role/content objects. We recover the original
    message structure here.
    """
    s = str(value or "").strip()
    if not s:
        return s

    if not (s.startswith("[") or s.startswith("{")):
        return s

    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def prompt_to_messages(value: Any) -> list[dict[str, str]]:
    """
    Convert a manifest/raw HealthBench prompt into canonical role/content
    messages while preserving all conversational turns.
    """
    if isinstance(value, str):
        parsed = _maybe_parse_structured_string(value)
        if parsed is not value and not isinstance(parsed, str):
            return prompt_to_messages(parsed)
        return [{"role": "user", "content": value}]

    if isinstance(value, dict):
        # Common single-message representation.
        if "role" in value and "content" in value:
            return [{
                "role": str(value.get("role") or "user").strip().lower(),
                "content": _flatten_text_content(value.get("content")),
            }]

        # Occasionally a wrapper object contains prompt/messages.
        for key in ("prompt", "messages", "conversation"):
            if key in value:
                return prompt_to_messages(value[key])

        return [{"role": "user", "content": json.dumps(value, ensure_ascii=False)}]

    if isinstance(value, list):
        out: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, dict) and ("role" in item or "content" in item):
                role = str(item.get("role") or "user").strip().lower()
                content = _flatten_text_content(item.get("content"))
                out.append({"role": role, "content": content})
            elif isinstance(item, str):
                out.append({"role": "user", "content": item})
            else:
                nested = prompt_to_messages(item)
                out.extend(nested)

        if out:
            return out

    return [{"role": "user", "content": str(value or "")}]


def canonicalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    allowed_roles = {"system", "user", "assistant"}
    out = []
    for m in messages:
        role = str(m.get("role") or "user").strip().lower()
        if role not in allowed_roles:
            role = "user"
        out.append({
            "role": role,
            "content": str(m.get("content") or ""),
        })
    if not out:
        raise ValueError("Conversation has no messages")
    return out


def append_safety_instruction(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Append the safety instruction to the LAST USER TURN, preserving the original
    HealthBench multi-turn conversation. This matches the intended BRIDGE input
    intervention more closely than serializing the whole conversation as one
    string.
    """
    out = copy.deepcopy(canonicalize_messages(messages))
    user_indices = [i for i, m in enumerate(out) if m["role"] == "user"]
    if not user_indices:
        raise ValueError("Cannot append safety instruction: no user turn")

    i = user_indices[-1]
    base = out[i]["content"].rstrip()
    out[i]["content"] = base + "\n\n" + SAFETY_INSTRUCTION
    return out


def messages_fingerprint(messages: list[dict[str, str]]) -> str:
    clean = [
        {
            "role": str(m.get("role") or "").strip().lower(),
            "content": norm(_flatten_text_content(m.get("content"))),
        }
        for m in canonicalize_messages(messages)
    ]
    return sha256_text(json.dumps(clean, ensure_ascii=False, sort_keys=True))


def looks_like_serialized_conversation_output(text: str) -> bool:
    """
    Detect an output that is itself a serialized role/content conversation.
    This is almost always a prompt-formatting failure rather than a valid
    assistant answer.
    """
    s = str(text or "").strip()
    if not (s.startswith("[") or s.startswith("{")):
        return False

    parsed = _maybe_parse_structured_string(s)
    if isinstance(parsed, dict):
        return (
            "role" in parsed
            and "content" in parsed
            and str(parsed.get("role", "")).lower() in {"user", "assistant", "system"}
        )

    if isinstance(parsed, list) and parsed:
        role_items = [
            x for x in parsed
            if isinstance(x, dict) and "role" in x and "content" in x
        ]
        return len(role_items) >= 1

    return False


def release_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Manifest
# -----------------------------------------------------------------------------


def load_manifest(path: Path):
    df = pd.read_csv(path)
    required = {"example_id", "source_line", "binary_label", "included", "split", "prompt"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Manifest missing columns: {sorted(missing)}")
    df = df.copy()
    df = df[df["included"].map(boolish)].copy()
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["example_id"] = df["example_id"].astype(str)
    df["binary_label"] = df["binary_label"].astype(int)
    return df


def split_rows(df: pd.DataFrame, split: str, limit: int, seed: int):
    x = df[df["split"] == split].copy().sort_values(["source_line", "example_id"], kind="stable")
    if not len(x):
        raise ValueError(f"No rows for split={split}")
    if limit and limit < len(x):
        x = x.sample(n=limit, random_state=seed).sort_values(["source_line", "example_id"], kind="stable")
    return x.reset_index(drop=True)


def train_rows(df: pd.DataFrame, limit: int, seed: int):
    x = df[df["split"] == "train"].copy()
    if not len(x):
        raise ValueError("No TRAIN rows")
    if limit == 0 or limit >= len(x):
        return x.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # only for smoke tests
    labels = sorted(x["binary_label"].astype(int).unique())
    if len(labels) != 2:
        return x.sample(n=limit, random_state=seed).reset_index(drop=True)
    idx = []
    each = limit // 2
    for y in labels:
        g = x[x["binary_label"].astype(int) == y]
        s = g.sample(n=min(each, len(g)), random_state=seed + int(y))
        idx.extend(s.index.tolist())
    need = limit - len(idx)
    if need > 0:
        rem = x.drop(index=idx, errors="ignore")
        if len(rem):
            idx.extend(rem.sample(n=min(need, len(rem)), random_state=seed + 99).index.tolist())
    return x.loc[idx].sample(frac=1.0, random_state=seed).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Frozen cached BRIDGE gate -- critical fix
# -----------------------------------------------------------------------------


def first_col(df: pd.DataFrame, names: list[str]):
    by_lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    return None


def load_probe_threshold(probe_dir: Path):
    meta_path = probe_dir / "selected_probe_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ["selected_threshold", "threshold", "validation_threshold", "probe_threshold"]:
            if key in meta:
                return float(meta[key])
    path = probe_dir / "threshold_validation.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    tc = [c for c in df.columns if "threshold" in str(c).lower()]
    bc = [c for c in df.columns if "balanced" in str(c).lower() and "accuracy" in str(c).lower()]
    if not tc:
        raise KeyError(f"No threshold column in {path}: {df.columns.tolist()}")
    if bc and len(df) > 1:
        idx = pd.to_numeric(df[bc[0]], errors="coerce").idxmax()
        return float(df.loc[idx, tc[0]])
    return float(df.iloc[0][tc[0]])


def detect_score_col(df: pd.DataFrame):
    preferred = [
        "evidence_score", "prob_insufficient", "probability_insufficient",
        "insufficient_probability", "positive_probability", "positive_prob",
        "probability", "prob", "y_score", "probe_score", "score",
    ]
    col = first_col(df, preferred)
    if col is not None:
        return col
    candidates = []
    for c in df.columns:
        name = str(c).lower()
        if any(k in name for k in ["label", "pred", "threshold", "source_line", "index", "layer", "id"]):
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().all() and len(vals):
            a = vals.to_numpy(float)
            if np.all(np.isfinite(a)) and np.all((a >= 0) & (a <= 1)):
                candidates.append(c)
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(f"Cannot identify probe score column. Columns={df.columns.tolist()}, candidates={candidates}")


def detect_pred_col(df: pd.DataFrame):
    return first_col(df, ["predicted_label", "prediction", "pred_label", "y_pred", "probe_prediction", "flagged", "pred"])


def detect_label_col(df: pd.DataFrame):
    return first_col(df, ["binary_label", "evidence_label", "true_label", "y_true", "label", "target"])


def align_probe_predictions(pred: pd.DataFrame, ev: pd.DataFrame):
    pairs = [
        ("example_id", "example_id"),
        ("prompt_id", "example_id"),
        ("manifest_id", "example_id"),
        ("id", "example_id"),
        ("prompt_hash", "prompt_hash"),
    ]
    for pc, ec in pairs:
        if pc in pred.columns and ec in ev.columns:
            p = pred.copy(); e = ev.copy()
            p["__k"] = p[pc].astype(str); e["__k"] = e[ec].astype(str)
            if not p["__k"].duplicated().any():
                lut = p.set_index("__k", drop=False)
                keys = e["__k"].tolist()
                if all(k in lut.index for k in keys):
                    print(f"Probe cache alignment: {pc} -> {ec}")
                    return lut.loc[keys].reset_index(drop=True)

    if all(c in pred.columns for c in ["source_file", "source_line"]) and all(c in ev.columns for c in ["source_file", "source_line"]):
        p = pred.copy(); e = ev.copy()
        p["__k"] = p["source_file"].astype(str) + "::" + pd.to_numeric(p["source_line"], errors="coerce").astype("Int64").astype(str)
        e["__k"] = e["source_file"].astype(str) + "::" + pd.to_numeric(e["source_line"], errors="coerce").astype("Int64").astype(str)
        if not p["__k"].duplicated().any():
            lut = p.set_index("__k", drop=False); keys = e["__k"].tolist()
            if all(k in lut.index for k in keys):
                print("Probe cache alignment: source_file + source_line")
                return lut.loc[keys].reset_index(drop=True)

    if "source_line" in pred.columns and "source_line" in ev.columns:
        p_lines = pd.to_numeric(pred["source_line"], errors="coerce")
        e_lines = pd.to_numeric(ev["source_line"], errors="coerce")
        if p_lines.notna().all() and e_lines.notna().all() and not p_lines.duplicated().any():
            p = pred.copy(); p["__line"] = p_lines.astype(int); lut = p.set_index("__line", drop=False)
            keys = e_lines.astype(int).tolist()
            if all(k in lut.index for k in keys):
                print("Probe cache alignment: source_line")
                return lut.loc[keys].reset_index(drop=True)

    # Last-resort row alignment is allowed only if all labels match exactly.
    label_col = detect_label_col(pred)
    if len(pred) == len(ev) and label_col is not None:
        labels = pd.to_numeric(pred[label_col], errors="coerce")
        if labels.notna().all() and np.array_equal(labels.astype(int).to_numpy(), ev["binary_label"].astype(int).to_numpy()):
            print("Probe cache alignment: row order (all labels match exactly)")
            return pred.reset_index(drop=True)

    raise RuntimeError("Could not safely align healthbench_test_predictions.csv to the test manifest")


def load_frozen_gate(probe_dir: Path, ev: pd.DataFrame):
    path = probe_dir / "healthbench_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required frozen probe cache missing: {path}")
    raw = pd.read_csv(path)
    aligned = align_probe_predictions(raw, ev)
    score_col = detect_score_col(aligned)
    pred_col = detect_pred_col(aligned)
    label_col = detect_label_col(aligned)
    threshold = load_probe_threshold(probe_dir)

    scores = pd.to_numeric(aligned[score_col], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores.to_numpy(float)).all():
        raise ValueError(f"Invalid frozen probe scores in {score_col}")
    scores = scores.to_numpy(float)

    if pred_col is not None:
        r = aligned[pred_col]
        n = pd.to_numeric(r, errors="coerce")
        flags = n.astype(int).astype(bool).to_numpy() if n.notna().all() else r.map(boolish).to_numpy(bool)
        flag_source = f"cached column {pred_col}"
    else:
        flags = scores >= threshold
        flag_source = "cached score + frozen threshold"

    y = ev["binary_label"].astype(int).to_numpy()
    if label_col is not None:
        cached_y = pd.to_numeric(aligned[label_col], errors="coerce")
        if cached_y.notna().all() and not np.array_equal(cached_y.astype(int).to_numpy(), y):
            raise RuntimeError("Cached probe labels do not match manifest labels")

    out = pd.DataFrame({
        "example_id": ev["example_id"].astype(str),
        "prompt_id": aligned["prompt_id"].astype(str) if "prompt_id" in aligned.columns else ev["example_id"].astype(str),
        "evidence_label": y,
        "evidence_score": scores,
        "flagged": flags.astype(bool),
        "probe_threshold": float(threshold),
    })

    auroc = roc_auc_score(y, scores)
    sens = recall_score(y, flags.astype(int), pos_label=1, zero_division=0)
    spec = recall_score(y, flags.astype(int), pos_label=0, zero_division=0)
    ba = balanced_accuracy_score(y, flags.astype(int))
    print("=" * 80)
    print("FROZEN CACHED BRIDGE GATE")
    print("file:", path)
    print("score column:", score_col)
    print("flag source:", flag_source)
    print("threshold:", threshold)
    print("n:", len(out), "flagged:", int(out["flagged"].sum()))
    print(f"AUROC={auroc:.6f} sensitivity={sens:.6f} specificity={spec:.6f} balanced_accuracy={ba:.6f}")
    print("=" * 80)
    return out


# -----------------------------------------------------------------------------
# Target model adapter
# -----------------------------------------------------------------------------


class TargetAdapter:
    """
    Architecture-aware target-model adapter.

    Critical formatting rule:
      inputs are rendered from the ORIGINAL HealthBench message list, not from a
      serialized Python/JSON representation of that list.
    """

    def __init__(
        self,
        model_key: str,
        path: Path,
        trust: bool = False,
        medgemma_internal_max_new_tokens: int = 2000,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")

        torch.cuda.set_device(0)
        self.device = torch.device("cuda:0")
        self.model_key = model_key
        self.model_path = Path(path)
        self.trust_remote_code = trust
        self.medgemma_internal_max_new_tokens = int(
            medgemma_internal_max_new_tokens
        )
        if self.medgemma_internal_max_new_tokens <= 0:
            raise ValueError(
                "medgemma_internal_max_new_tokens must be positive"
            )

        cfg = AutoConfig.from_pretrained(
            path,
            trust_remote_code=trust,
        )
        self.model_type = str(getattr(cfg, "model_type", ""))
        self.dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        if self.model_type == "gemma3":
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration

            self.kind = "processor"
            self.front = AutoProcessor.from_pretrained(
                path,
                trust_remote_code=trust,
            )
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                path,
                dtype=self.dtype,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                trust_remote_code=trust,
            )

        elif self.model_type in {"qwen2_5_vl", "qwen2.5_vl"}:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            self.kind = "processor"
            self.front = AutoProcessor.from_pretrained(
                path,
                trust_remote_code=trust,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path,
                dtype=self.dtype,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                trust_remote_code=trust,
            )

        else:
            self.kind = "tokenizer"
            self.front = AutoTokenizer.from_pretrained(
                path,
                trust_remote_code=trust,
                use_fast=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                dtype=self.dtype,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                trust_remote_code=trust,
            )

        self.model.eval()

        if self.tok.pad_token_id is None:
            if self.tok.eos_token_id is None:
                raise RuntimeError("Target tokenizer has neither PAD nor EOS token")
            self.tok.pad_token = self.tok.eos_token

        self.tok.padding_side = "left"
        self.layers_name, self.layers = self._layers()

        print("=" * 80)
        print("TARGET MODEL")
        print("=" * 80)
        print("model_key:", self.model_key)
        print("class:", self.model.__class__.__name__)
        print("model_type:", self.model_type)
        print("decoder layers:", self.layers_name, len(self.layers))
        print("dtype:", self.dtype)
        print("=" * 80)

    @property
    def tok(self):
        return self.front if self.kind == "tokenizer" else self.front.tokenizer

    def _layers(self):
        expected = None

        for cfg in [
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "config", None), "text_config", None),
        ]:
            if cfg is not None and getattr(cfg, "num_hidden_layers", None) is not None:
                expected = int(cfg.num_hidden_layers)
                break

        cand = [
            (n, m)
            for n, m in self.model.named_modules()
            if isinstance(m, torch.nn.ModuleList) and n.endswith("layers")
        ]

        exact = [
            x for x in cand
            if expected is not None and len(x[1]) == expected
        ]

        if exact:
            exact.sort(
                key=lambda x: (
                    0
                    if (
                        "language" in x[0]
                        or "text" in x[0]
                        or x[0] == "model.layers"
                    )
                    else 1,
                    len(x[0]),
                )
            )
            return exact[0]

        if cand:
            return sorted(cand, key=lambda x: len(x[1]), reverse=True)[0]

        raise RuntimeError("Could not locate decoder layers")

    @staticmethod
    def _llama3_manual_render(messages: list[dict[str, str]]) -> str:
        """
        OpenBioLLM uses a Llama-3 tokenizer without a configured chat_template
        in the canonical BRIDGE environment, so reconstruct the complete
        multi-turn Llama-3 conversation explicitly.
        """
        text = "<|begin_of_text|>"

        for message in canonicalize_messages(messages):
            role = message["role"]
            content = message["content"]
            text += (
                f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                f"{content}"
                "<|eot_id|>"
            )

        text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return text

    def _processor_messages(self, messages: list[dict[str, str]]):
        """
        Gemma3 / Qwen2.5-VL processors accept text-only multimodal-style
        content blocks.
        """
        return [
            {
                "role": m["role"],
                "content": [{"type": "text", "text": m["content"]}],
            }
            for m in canonicalize_messages(messages)
        ]

    def render(self, messages: list[dict[str, str]]) -> str:
        messages = canonicalize_messages(messages)

        if self.model_key == "openbiollm":
            return self._llama3_manual_render(messages)

        if self.kind == "processor":
            proc_messages = self._processor_messages(messages)
            return self.front.apply_chat_template(
                proc_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        raise RuntimeError(
            f"No safe canonical prompt-format path for {self.model_key}"
        )

    def encode(self, messages: list[dict[str, str]]):
        messages = canonicalize_messages(messages)

        if self.model_key == "medgemma":
            # Match the official MedGemma Transformers usage: the processor
            # applies the chat template and tokenizes in one operation.
            proc_messages = self._processor_messages(messages)
            b = self.front.apply_chat_template(
                proc_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

        elif self.kind == "processor":
            text = self.render(messages)
            b = self.front(
                text=[text],
                images=None,
                return_tensors="pt",
                padding=True,
            )

        else:
            text = self.render(messages)
            b = self.front(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            )

        b = {
            k: v.to(self.device)
            for k, v in b.items()
            if torch.is_tensor(v)
        }

        if "input_ids" not in b:
            raise RuntimeError(f"No input_ids; keys={list(b)}")

        return b

    def eos_ids(self):
        if self.model_key == "openbiollm":
            eot = self.tok.convert_tokens_to_ids("<|eot_id|>")
            ids = [self.tok.eos_token_id, eot]
            ids = [
                int(x)
                for x in ids
                if x is not None and int(x) >= 0
            ]
            return list(dict.fromkeys(ids))

        if self.model_key == "medgemma":
            ids = getattr(
                getattr(self.model, "generation_config", None),
                "eos_token_id",
                None,
            )
            if ids is not None:
                if isinstance(ids, (list, tuple)):
                    return [int(x) for x in ids]
                return int(ids)

        return self.tok.eos_token_id

    @staticmethod
    def hidden(out):
        if torch.is_tensor(out):
            return out
        if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
            return out[0]
        raise TypeError(type(out))

    @staticmethod
    def replace(out, h):
        if torch.is_tensor(out):
            return h
        if isinstance(out, tuple):
            return (h,) + tuple(out[1:])
        raise TypeError(type(out))

    @torch.inference_mode()
    def activations(
        self,
        messages: list[dict[str, str]],
        indices: Iterable[int],
    ):
        got = {}
        handles = []

        requested = sorted(set(int(i) for i in indices))

        for idx in requested:
            if not 0 <= idx < len(self.layers):
                raise IndexError(idx)

            def make_hook(j):
                def hook(_m, _i, o):
                    got[j] = (
                        self.hidden(o)[:, -1, :]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()[0]
                    )
                    return o
                return hook

            handles.append(
                self.layers[idx].register_forward_hook(
                    make_hook(idx)
                )
            )

        try:
            self.model(
                **self.encode(messages),
                use_cache=False,
                return_dict=True,
            )
        finally:
            for h in handles:
                h.remove()

        missing = [
            i for i in requested
            if i not in got
        ]
        if missing:
            raise RuntimeError(f"Hooks did not fire: {missing}")

        for idx, arr in got.items():
            if not np.isfinite(arr).all():
                raise RuntimeError(
                    f"Non-finite activation at layer {idx}"
                )

        return got

    def _truncate_user_facing_answer(
        self,
        text: str,
        max_visible_tokens: int,
    ) -> str:
        text = str(text or "").strip()

        if not text:
            return text

        ids = self.tok(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        if ids and isinstance(ids[0], list):
            ids = ids[0]

        if len(ids) <= max_visible_tokens:
            return text

        return self.tok.decode(
            ids[:max_visible_tokens],
            skip_special_tokens=True,
        ).strip()

    def _clean_generation(
        self,
        raw_text: str,
        max_visible_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        text = str(raw_text or "").strip()

        info = {
            "generation_failure": False,
            "generation_status": "ok",
            "thought_trace_present": False,
            "final_marker_found": False,
        }

        if self.model_key == "medgemma":
            # MedGemma may emit:
            #   <unused94>thought ... <unused95>USER-FACING ANSWER
            #
            # IMPORTANT: if the model never reaches <unused95> within the
            # pre-specified internal budget, this is treated as an observed
            # generation failure. We do NOT keep increasing the budget until
            # an answer appears, and we do NOT expose/grade the private
            # reasoning trace. The user-facing response is therefore empty.
            start_match = re.search(
                r"<unused\s*94>\s*thought\s*",
                text,
                flags=re.I,
            )
            end_match = re.search(
                r"<unused\s*95>",
                text,
                flags=re.I,
            )

            info["thought_trace_present"] = start_match is not None
            info["final_marker_found"] = end_match is not None

            if start_match is not None:
                if (
                    end_match is None
                    or end_match.start() <= start_match.start()
                ):
                    info["generation_failure"] = True
                    info["generation_status"] = "no_final_within_budget"
                    return "", info

                text = text[end_match.end():].strip()

            elif end_match is not None:
                text = text[end_match.end():].strip()

            text = re.sub(
                r"<(?:bos|eos|pad|start_of_turn|end_of_turn)>",
                "",
                text,
                flags=re.I,
            ).strip()

            text = re.sub(
                r"^\s*model\s*\n",
                "",
                text,
                flags=re.I,
            ).strip()

            if not text:
                info["generation_failure"] = True
                info["generation_status"] = "empty_user_facing_answer"
                return "", info

            text = self._truncate_user_facing_answer(
                text,
                max_visible_tokens,
            )

        return text, info

    @torch.inference_mode()
    def generate(
        self,
        messages: list[dict[str, str]],
        max_new: int,
        layer: int | None = None,
        delta: np.ndarray | None = None,
    ):
        b = self.encode(messages)
        width = int(b["input_ids"].shape[1])

        handle = None
        stats = {
            "hook_fired": False,
            "delta_norm": 0.0,
        }

        if layer is not None and delta is not None:
            d = torch.tensor(
                np.asarray(delta, np.float32),
                device=self.device,
            )
            done = {"x": False}

            def hook(_m, _i, o):
                h = self.hidden(o)

                if (
                    not done["x"]
                    and h.ndim == 3
                    and h.shape[1] == width
                ):
                    if h.shape[-1] != d.numel():
                        raise RuntimeError("Steering dimension mismatch")

                    nh = h.clone()
                    dd = d.to(nh.dtype)
                    nh[:, -1, :] += dd
                    done["x"] = True

                    stats["hook_fired"] = True
                    stats["delta_norm"] = float(
                        torch.linalg.vector_norm(
                            dd.float()
                        ).item()
                    )

                    return self.replace(o, nh)

                return o

            handle = self.layers[layer].register_forward_hook(hook)

        generation_budget = int(max_new)

        if self.model_key == "medgemma":
            generation_budget = max(
                generation_budget,
                self.medgemma_internal_max_new_tokens,
            )

        try:
            g = self.model.generate(
                **b,
                max_new_tokens=generation_budget,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.eos_ids(),
            )
        finally:
            if handle:
                handle.remove()

        generated_ids = g[0, width:]

        # Preserve MedGemma's <unused94>/<unused95> delimiters until after
        # extracting the user-facing final answer.
        raw_text = self.tok.decode(
            generated_ids,
            skip_special_tokens=(self.model_key != "medgemma"),
        ).strip()

        text, clean_info = self._clean_generation(
            raw_text,
            max_visible_tokens=int(max_new),
        )

        stats.update(clean_info)
        stats["generation_token_budget"] = int(generation_budget)
        stats["raw_generated_tokens"] = int(generated_ids.numel())
        stats["raw_generation_sha256"] = sha256_text(raw_text)

        return text, stats

    def close(self):
        self.layers = None
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.front
        except Exception:
            pass
        release_cuda()


# -----------------------------------------------------------------------------
# M2 controller
# -----------------------------------------------------------------------------


def fit_controller(
    adapter: TargetAdapter,
    train_df: pd.DataFrame,
    layer: int,
    rank: int,
    lam: float,
    outdir: Path,
):
    """
    Fit the M2 actuator on TRAIN only.

    The base and safety-conditioned activations use the same canonical
    HealthBench conversation; the safety instruction is appended only to the
    final user turn.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    H = []
    D = []

    for i, (_, row) in enumerate(train_df.iterrows(), start=1):
        messages = prompt_to_messages(row["prompt"])
        safe_messages = append_safety_instruction(messages)

        h_base = adapter.activations(
            messages,
            [layer],
        )[layer]

        h_safe = adapter.activations(
            safe_messages,
            [layer],
        )[layer]

        H.append(h_base)
        D.append(h_safe - h_base)

        if i == 1 or i % 10 == 0 or i == len(train_df):
            print(
                f"[behavior pairs] {i}/{len(train_df)}",
                flush=True,
            )

    H = np.vstack(H).astype(np.float32)
    D = np.vstack(D).astype(np.float32)

    if not np.isfinite(H).all() or not np.isfinite(D).all():
        raise RuntimeError("Non-finite controller activations")

    if rank > min(D.shape):
        raise ValueError(
            f"rank={rank} too large for D={D.shape}"
        )

    # Uncentered SVD preserves the average safety-instruction shift.
    _, singular_values, Vt = np.linalg.svd(
        D.astype(np.float64),
        full_matrices=False,
    )

    B = Vt[:rank].astype(np.float32)

    mean = H.mean(axis=0).astype(np.float32)
    scale = H.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0

    X = ((H - mean) / scale).astype(np.float32)
    C = (D @ B.T).astype(np.float32)

    reg = Ridge(
        alpha=lam,
        fit_intercept=True,
        solver="lsqr",
    ).fit(X, C)

    np.save(
        outdir / "behavior_basis.npy",
        B,
    )
    np.save(
        outdir / "x_mean.npy",
        mean,
    )
    np.save(
        outdir / "x_scale.npy",
        scale,
    )
    np.save(
        outdir / "singular_values.npy",
        singular_values.astype(np.float32),
    )
    joblib.dump(
        reg,
        outdir / f"ridge_rank{rank}.joblib",
    )

    energy = singular_values ** 2
    frac = (
        float(energy[:rank].sum() / energy.sum())
        if energy.sum()
        else 0.0
    )

    train_pred = reg.predict(X)
    train_mse = float(
        np.mean(
            (train_pred - C) ** 2
        )
    )

    meta = {
        "n_train_examples": int(len(H)),
        "hidden_dim": int(H.shape[1]),
        "steering_layer_0based": int(layer),
        "steering_layer_1based": int(layer + 1),
        "rank": int(rank),
        "ridge_lambda": float(lam),
        "cumulative_behavior_energy": frac,
        "train_coefficient_mse": train_mse,
        "prompt_policy": (
            "canonical HealthBench messages; safety instruction appended "
            "to final user turn"
        ),
        "safety_instruction": SAFETY_INSTRUCTION,
    }

    (
        outdir
        / "behavior_metadata.json"
    ).write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    return B, mean, scale, reg


def load_controller(outdir: Path, rank: int):
    B = np.load(
        outdir / "behavior_basis.npy"
    ).astype(np.float32)

    mean = np.load(
        outdir / "x_mean.npy"
    ).astype(np.float32)

    scale = np.load(
        outdir / "x_scale.npy"
    ).astype(np.float32)

    reg = joblib.load(
        outdir / f"ridge_rank{rank}.joblib"
    )

    if B.shape[0] != rank:
        raise RuntimeError(
            f"Saved basis rank={B.shape[0]} but expected {rank}"
        )

    return B, mean, scale, reg


def predict_delta(h, B, mean, scale, reg):
    x = (
        np.asarray(h, np.float32)
        - mean
    ) / scale

    c = reg.predict(
        x.reshape(1, -1)
    )[0]

    return np.asarray(
        c @ B,
        np.float32,
    )


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------


def generate_m2(args, manifest: pd.DataFrame):
    ev_full = split_rows(
        manifest,
        "test",
        0,
        args.seed,
    )

    if len(ev_full) != 188 and args.limit == 0:
        raise RuntimeError(
            f"Expected 188 HealthBench Hard prompts, found {len(ev_full)}"
        )

    gate_full = load_frozen_gate(
        args.probe_root / args.model_key,
        ev_full,
    )

    if args.limit and args.limit < len(ev_full):
        chosen = (
            ev_full.sample(
                n=args.limit,
                random_state=args.seed,
            )
            .sort_values(
                ["source_line", "example_id"],
                kind="stable",
            )
        )

        idx = chosen.index.tolist()

        ev = (
            ev_full.loc[idx]
            .reset_index(drop=True)
        )

        gate_df = (
            gate_full.loc[idx]
            .reset_index(drop=True)
        )

    else:
        ev = ev_full.reset_index(drop=True)
        gate_df = gate_full.reset_index(drop=True)

    if not np.array_equal(
        ev["example_id"].astype(str).to_numpy(),
        gate_df["example_id"].astype(str).to_numpy(),
    ):
        raise RuntimeError("Gate/example alignment mismatch")

    if args.gate_floor != 1.0:
        raise ValueError(
            "Final corrected M2 requires --gate-floor 1"
        )

    cond = (
        f"m2_adaptive_rank{args.rank}_alpha{args.alpha:g}"
    )

    out = (
        args.output_dir
        / "m2_generations.jsonl"
    )

    art = (
        args.output_dir
        / "behavior_artifacts"
    )

    if args.force_generate and out.exists():
        out.unlink()

    existing_rows = (
        read_jsonl(out)
        if out.exists()
        else []
    )

    existing = {
        (
            str(r.get("example_id")),
            str(r.get("condition")),
        )
        for r in existing_rows
    }

    wanted = {
        (str(e), c)
        for e in ev["example_id"]
        for c in [
            "no_intervention",
            cond,
        ]
    }

    if wanted.issubset(existing):
        print(
            "Generation cache complete; skipping target model"
        )
        return out

    adapter = TargetAdapter(
        args.model_key,
        args.model_path,
        args.trust_remote_code,
        medgemma_internal_max_new_tokens=(
            args.medgemma_internal_max_new_tokens
        ),
    )

    steer = (
        args.behavior_steering_layer_1based
        - 1
    )

    if not 0 <= steer < len(adapter.layers):
        raise IndexError("Invalid steering layer")

    tr = train_rows(
        manifest,
        args.behavior_train_limit,
        args.behavior_train_seed,
    )

    controller_files = [
        art / "behavior_basis.npy",
        art / "x_mean.npy",
        art / "x_scale.npy",
        art / f"ridge_rank{args.rank}.joblib",
    ]

    if (
        all(p.exists() for p in controller_files)
        and not args.force_generate
    ):
        B, mean, scale, reg = load_controller(
            art,
            args.rank,
        )
    else:
        B, mean, scale, reg = fit_controller(
            adapter,
            tr,
            steer,
            args.rank,
            args.ridge_lambda,
            art,
        )

    baseline_cache = {
        str(r["example_id"]): str(r["response_text"])
        for r in existing_rows
        if r.get("condition") == "no_intervention"
    }

    baseline_stats_cache = {
        str(r["example_id"]): {
            "generation_failure": boolish(
                r.get("generation_failure", False)
            ),
            "generation_status": str(
                r.get("generation_status", "ok")
            ),
            "thought_trace_present": boolish(
                r.get("thought_trace_present", False)
            ),
            "final_marker_found": boolish(
                r.get("final_marker_found", False)
            ),
            "generation_token_budget": r.get(
                "generation_token_budget"
            ),
            "raw_generated_tokens": r.get(
                "raw_generated_tokens"
            ),
            "raw_generation_sha256": r.get(
                "raw_generation_sha256"
            ),
        }
        for r in existing_rows
        if r.get("condition") == "no_intervention"
    }

    for i, (_, mr) in enumerate(
        ev.iterrows(),
        start=1,
    ):
        g = gate_df.iloc[i - 1]

        ex = str(mr["example_id"])
        y = int(mr["binary_label"])
        messages = prompt_to_messages(
            mr["prompt"]
        )

        pid = str(
            g["prompt_id"]
        )

        score = float(
            g["evidence_score"]
        )

        tau = float(
            g["probe_threshold"]
        )

        flagged = bool(
            g["flagged"]
        )

        print(
            f"[{i}/{len(ev)}] {ex} "
            f"y={y} score={score:.6f} flagged={int(flagged)} "
            f"turns={len(messages)}",
            flush=True,
        )

        prompt_fingerprint = messages_fingerprint(
            messages
        )

        bkey = (
            ex,
            "no_intervention",
        )

        if bkey not in existing:
            baseline_text, baseline_stats = adapter.generate(
                messages,
                args.max_new_tokens,
            )

            append_jsonl(
                out,
                {
                    "prompt_id": pid,
                    "example_id": ex,
                    "model": args.model_key,
                    "condition": "no_intervention",
                    "response_text": baseline_text,
                    "evidence_label": y,
                    "evidence_score": score,
                    "probe_threshold": tau,
                    "flagged": flagged,
                    "gate_weight": (
                        1.0
                        if flagged
                        else 0.0
                    ),
                    "rank": None,
                    "alpha": None,
                    "effective_alpha": 0.0,
                    "direction_kind": "none",
                    "experiment": "M2",
                    "layer_0based": steer,
                    "layer_1based": steer + 1,
                    "intervention_scope": (
                        "prefill_last_prompt_token"
                    ),
                    "raw_predicted_delta_norm": 0.0,
                    "applied_delta_norm": 0.0,
                    "hook_fired": False,
                    "generation_failure": bool(
                        baseline_stats.get("generation_failure", False)
                    ),
                    "generation_status": str(
                        baseline_stats.get("generation_status", "ok")
                    ),
                    "thought_trace_present": bool(
                        baseline_stats.get("thought_trace_present", False)
                    ),
                    "final_marker_found": bool(
                        baseline_stats.get("final_marker_found", False)
                    ),
                    "generation_token_budget": baseline_stats.get(
                        "generation_token_budget"
                    ),
                    "raw_generated_tokens": baseline_stats.get(
                        "raw_generated_tokens"
                    ),
                    "raw_generation_sha256": baseline_stats.get(
                        "raw_generation_sha256"
                    ),
                    "split": "test",
                    "gate_source": str(
                        (
                            args.probe_root
                            / args.model_key
                            / "healthbench_test_predictions.csv"
                        ).resolve()
                    ),
                    "source_file": mr.get(
                        "source_file"
                    ),
                    "source_line": (
                        int(mr["source_line"])
                        if pd.notna(mr["source_line"])
                        else None
                    ),
                    "prompt_sha256": prompt_fingerprint,
                    "n_prompt_messages": len(messages),
                    "prompt_format_policy": (
                        "canonical_healthbench_messages"
                    ),
                },
            )

            existing.add(
                bkey
            )

            baseline_cache[
                ex
            ] = baseline_text

            baseline_stats_cache[
                ex
            ] = baseline_stats

        else:
            baseline_text = baseline_cache[
                ex
            ]

            baseline_stats = baseline_stats_cache.get(
                ex,
                {
                    "generation_failure": False,
                    "generation_status": "ok",
                    "thought_trace_present": False,
                    "final_marker_found": False,
                    "generation_token_budget": None,
                    "raw_generated_tokens": None,
                    "raw_generation_sha256": None,
                },
            )

        mkey = (
            ex,
            cond,
        )

        if mkey in existing:
            continue

        if not flagged:
            # Exact binary gate is off. Under deterministic decoding the M2
            # condition must equal baseline exactly.
            generated_text = baseline_text

            raw_norm = 0.0
            effective_alpha = 0.0

            stats = {
                "hook_fired": False,
                "delta_norm": 0.0,
                "generation_failure": bool(
                    baseline_stats.get("generation_failure", False)
                ),
                "generation_status": str(
                    baseline_stats.get("generation_status", "ok")
                ),
                "thought_trace_present": bool(
                    baseline_stats.get("thought_trace_present", False)
                ),
                "final_marker_found": bool(
                    baseline_stats.get("final_marker_found", False)
                ),
                "generation_token_budget": baseline_stats.get(
                    "generation_token_budget"
                ),
                "raw_generated_tokens": baseline_stats.get(
                    "raw_generated_tokens"
                ),
                "raw_generation_sha256": baseline_stats.get(
                    "raw_generation_sha256"
                ),
            }

        else:
            h = adapter.activations(
                messages,
                [steer],
            )[steer]

            raw = predict_delta(
                h,
                B,
                mean,
                scale,
                reg,
            )

            raw_norm = float(
                np.linalg.norm(raw)
            )

            effective_alpha = float(
                args.alpha
            )

            generated_text, stats = adapter.generate(
                messages,
                args.max_new_tokens,
                steer,
                raw * effective_alpha,
            )

            if not stats["hook_fired"]:
                raise RuntimeError(
                    f"M2 hook failed for {ex}"
                )

        append_jsonl(
            out,
            {
                "prompt_id": pid,
                "example_id": ex,
                "model": args.model_key,
                "condition": cond,
                "response_text": generated_text,
                "evidence_label": y,
                "evidence_score": score,
                "probe_threshold": tau,
                "flagged": flagged,
                "gate_weight": (
                    1.0
                    if flagged
                    else 0.0
                ),
                "rank": int(args.rank),
                "alpha": float(args.alpha),
                "effective_alpha": effective_alpha,
                "direction_kind": (
                    "adaptive_behavior_subspace"
                ),
                "experiment": "M2",
                "layer_0based": steer,
                "layer_1based": steer + 1,
                "intervention_scope": (
                    "prefill_last_prompt_token"
                ),
                "raw_predicted_delta_norm": raw_norm,
                "applied_delta_norm": float(
                    stats["delta_norm"]
                ),
                "hook_fired": bool(
                    stats["hook_fired"]
                ),
                "generation_failure": bool(
                    stats.get("generation_failure", False)
                ),
                "generation_status": str(
                    stats.get("generation_status", "ok")
                ),
                "thought_trace_present": bool(
                    stats.get("thought_trace_present", False)
                ),
                "final_marker_found": bool(
                    stats.get("final_marker_found", False)
                ),
                "generation_token_budget": stats.get(
                    "generation_token_budget"
                ),
                "raw_generated_tokens": stats.get(
                    "raw_generated_tokens"
                ),
                "raw_generation_sha256": stats.get(
                    "raw_generation_sha256"
                ),
                "split": "test",
                "gate_source": str(
                    (
                        args.probe_root
                        / args.model_key
                        / "healthbench_test_predictions.csv"
                    ).resolve()
                ),
                "source_file": mr.get(
                    "source_file"
                ),
                "source_line": (
                    int(mr["source_line"])
                    if pd.notna(mr["source_line"])
                    else None
                ),
                "prompt_sha256": prompt_fingerprint,
                "n_prompt_messages": len(messages),
                "prompt_format_policy": (
                    "canonical_healthbench_messages"
                ),
            },
        )

        existing.add(
            mkey
        )

    # -------------------------------------------------------------------------
    # Mechanical audit
    # -------------------------------------------------------------------------

    rows = [
        r
        for r in read_jsonl(out)
        if r.get("condition")
        in {
            "no_intervention",
            cond,
        }
    ]

    df = pd.DataFrame(
        rows
    )

    if len(df) != 2 * len(ev):
        raise RuntimeError(
            f"Expected {2 * len(ev)} rows, got {len(df)}"
        )

    if (
        df[
            ["example_id", "condition"]
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate example/condition rows in generation file"
        )

    m2 = df[
        df["condition"]
        == cond
    ].copy()

    flagged_m2 = m2[
        m2["flagged"].map(boolish)
    ]

    unflagged_m2 = m2[
        ~m2["flagged"].map(boolish)
    ]

    if (
        len(flagged_m2)
        and not flagged_m2[
            "hook_fired"
        ].map(boolish).all()
    ):
        raise RuntimeError(
            "Flagged M2 row without hook firing"
        )

    if (
        len(unflagged_m2)
        and unflagged_m2[
            "hook_fired"
        ].map(boolish).any()
    ):
        raise RuntimeError(
            "Unflagged M2 row unexpectedly fired hook"
        )

    base = (
        df[
            df["condition"]
            == "no_intervention"
        ]
        .set_index(
            "example_id"
        )[
            "response_text"
        ]
        .to_dict()
    )

    for _, r in unflagged_m2.iterrows():
        if str(r["response_text"]) != str(
            base[str(r["example_id"])]
        ):
            raise RuntimeError(
                f"Unflagged output changed for {r['example_id']}"
            )

    # Detect the exact malformed pattern that contaminated the previous run.
    serialized_rows = [
        r
        for r in rows
        if looks_like_serialized_conversation_output(
            str(r.get("response_text", ""))
        )
    ]

    print()
    print("=" * 80)
    print("GENERATION FORMAT AUDIT")
    print("=" * 80)
    print(
        "Serialized role/content conversation outputs:",
        len(serialized_rows),
        "/",
        len(rows),
    )

    # Allow only a very small number of unusual outputs. A large count means
    # the model was prompted with serialized conversation syntax.
    if len(serialized_rows) > 3:
        examples = [
            {
                "example_id": r.get("example_id"),
                "condition": r.get("condition"),
                "preview": str(r.get("response_text", ""))[:180],
            }
            for r in serialized_rows[:5]
        ]

        raise RuntimeError(
            "Generation formatting audit failed: "
            f"{len(serialized_rows)} responses are serialized role/content "
            f"conversations. Examples={examples}"
        )

    print("Generation format audit: PASS")
    print("=" * 80)

    failure_df = df[
        df.get(
            "generation_failure",
            pd.Series(False, index=df.index),
        ).map(boolish)
    ].copy()

    print()
    print("=" * 80)
    print("GENERATION FAILURE AUDIT")
    print("=" * 80)

    if len(failure_df):
        failure_counts = (
            failure_df.groupby("condition")
            .size()
            .to_dict()
        )
    else:
        failure_counts = {}

    print(
        "No-user-facing-answer failures:",
        int(len(failure_df)),
        "/",
        int(len(df)),
    )
    print("By condition:", failure_counts)
    print("=" * 80)

    failure_df.to_csv(
        args.output_dir / "generation_failures.csv",
        index=False,
    )

    meta = {
        "experiment": "M2_FINAL",
        "model_key": args.model_key,
        "model_class": adapter.model.__class__.__name__,
        "model_type": adapter.model_type,
        "decoder_layers_path": adapter.layers_name,
        "eval_split": "test",
        "n_eval_examples": int(len(ev)),
        "gate_source": str(
            (
                args.probe_root
                / args.model_key
                / "healthbench_test_predictions.csv"
            ).resolve()
        ),
        "gate_mode": "binary_cached",
        "behavior_train_examples": int(len(tr)),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "ridge_lambda": float(args.ridge_lambda),
        "behavior_steering_layer_0based": int(steer),
        "behavior_steering_layer_1based": int(steer + 1),
        "max_new_tokens": int(args.max_new_tokens),
        "medgemma_internal_max_new_tokens": int(
            args.medgemma_internal_max_new_tokens
        ),
        "decoding": "greedy",
        "intervention_scope": (
            "prefill_last_prompt_token"
        ),
        "prompt_format_policy": (
            "canonical HealthBench role/content messages; "
            "serialized manifest conversations are parsed back to messages"
        ),
        "safety_instruction_policy": (
            "append to final user turn"
        ),
        "medgemma_thinking_policy": (
            "native MedGemma generation; allow built-in thought trace to "
            "finish; strip <unused94>...<unused95>; cap final user-facing "
            "answer to max_new_tokens"
        ),
        "serialized_conversation_outputs": int(
            len(serialized_rows)
        ),
        "generation_failures_total": int(
            len(failure_df)
        ),
        "generation_failures_by_condition": {
            str(k): int(v)
            for k, v in failure_counts.items()
        },
        "medgemma_failure_policy": (
            "If a thought trace does not reach <unused95> within the "
            "pre-specified internal budget, record an empty user-facing "
            "response and count it as a generation failure; do not exclude "
            "the example and do not increase the budget post hoc."
        ),
        "leakage_control": (
            "controller TRAIN-only; cached validated test gate; "
            "test labels never control intervention"
        ),
    }

    (
        args.output_dir
        / "generation_metadata.json"
    ).write_text(
        json.dumps(
            meta,
            indent=2,
        ),
        encoding="utf-8",
    )

    adapter.close()

    print("Saved", out)

    return out


# -----------------------------------------------------------------------------
# SAME HEALTHBENCH RUBRIC EVALUATION POLICY AS THE CAUSAL EXPERIMENTS
# -----------------------------------------------------------------------------

M2_EVAL_FIELDS = [
    "example_id",
    "evidence_label",
    "evidence_score",
    "probe_threshold",
    "flagged",
    "gate_weight",
    "rank",
    "alpha",
    "effective_alpha",
    "direction_kind",
    "experiment",
    "layer_0based",
    "layer_1based",
    "intervention_scope",
    "raw_predicted_delta_norm",
    "applied_delta_norm",
    "hook_fired",
    "split",
    "gate_source",
    "source_file",
    "source_line",
    "prompt_sha256",
    "n_prompt_messages",
    "prompt_format_policy",
    "generation_failure",
    "generation_status",
    "thought_trace_present",
    "final_marker_found",
    "generation_token_budget",
    "raw_generated_tokens",
    "raw_generation_sha256",
]


def find_healthbench_eval_py(repo: Path) -> Path:
    repo = repo.resolve()
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


def load_healthbench_reference(repo: Path):
    """
    Import the local OpenAI simple-evals HealthBench evaluator.

    We deliberately use its:
      * GRADER_TEMPLATE
      * RubricItem
      * parse_json_to_dict
      * calculate_score

    This is the same scoring policy used for the causal-experiment evaluator.
    """
    hb_file = find_healthbench_eval_py(repo)
    package_dir = hb_file.parent

    package_name = "simple_evals_m2_causal_policy"
    for name in list(sys.modules):
        if name == package_name or name.startswith(package_name + "."):
            del sys.modules[name]

    pkg = types.ModuleType(package_name)
    pkg.__path__ = [str(package_dir)]
    pkg.__package__ = package_name
    pkg.__spec__ = importlib.machinery.ModuleSpec(
        package_name, loader=None, is_package=True
    )
    sys.modules[package_name] = pkg

    hb = importlib.import_module(f"{package_name}.healthbench_eval")
    return hb, hb_file


class LocalHealthBenchJudge:
    """
    Local LLM rubric judge using the same policy as the causal evaluator.

    The model is asked to grade EACH HealthBench rubric item with:
        criteria_met: true / false

    This class does NOT classify ASK / ABSTAIN / DIRECT.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int,
        max_new_tokens: int,
        max_retries: int,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the local HealthBench judge")

        torch.cuda.set_device(0)
        self.device = torch.device("cuda:0")
        self.batch_size = int(batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.max_retries = int(max_retries)
        self.model_name = str(model_name)

        print("=" * 80)
        print("LOADING LOCAL HEALTHBENCH RUBRIC JUDGE")
        print("=" * 80)
        print("Judge:", self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise RuntimeError("Judge tokenizer has neither PAD nor EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map={"": 0},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        print("Judge class:", self.model.__class__.__name__)
        print("dtype:", dtype)
        print("GPU:", torch.cuda.get_device_name(0))
        print("=" * 80)

    def format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "USER:\n" + prompt + "\n\nASSISTANT:\n"

    @staticmethod
    def fallback_parse(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        m = re.search(r"\{.*\}", cleaned, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        lowered = cleaned.lower()
        if re.search(r'"?criteria_met"?\s*:\s*true', lowered):
            return {"criteria_met": True, "explanation": cleaned}
        if re.search(r'"?criteria_met"?\s*:\s*false', lowered):
            return {"criteria_met": False, "explanation": cleaned}
        return {}

    @torch.inference_mode()
    def _generate_batch(self, prompts: list[str]) -> list[str]:
        rendered = [self.format_prompt(p) for p in prompts]

        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        prompt_width = int(encoded["input_ids"].shape[1])

        out = self.model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return [
            self.tokenizer.decode(
                row[prompt_width:],
                skip_special_tokens=True,
            ).strip()
            for row in out
        ]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """
        Recursively split only if a batch-level CUDA OOM occurs.
        """
        try:
            return self._generate_batch(prompts)
        except torch.OutOfMemoryError:
            if len(prompts) == 1:
                raise
            torch.cuda.empty_cache()
            mid = len(prompts) // 2
            return (
                self.generate_batch(prompts[:mid])
                + self.generate_batch(prompts[mid:])
            )

    def grade(self, prompts: list[str], reference_parser):
        results: list[dict[str, Any] | None] = [None] * len(prompts)
        pending = list(range(len(prompts)))
        attempt = 0

        while pending:
            current = pending
            pending = []

            for start in range(0, len(current), self.batch_size):
                indices = current[start : start + self.batch_size]
                batch_prompts = [prompts[i] for i in indices]

                if attempt > 0:
                    batch_prompts = [
                        p
                        + "\n\nReturn exactly one valid JSON object with "
                          "criteria_met as true or false."
                        for p in batch_prompts
                    ]

                outputs = self.generate_batch(batch_prompts)

                for idx, raw in zip(indices, outputs, strict=True):
                    try:
                        parsed = reference_parser(raw)
                    except Exception:
                        parsed = {}

                    if not isinstance(parsed, dict):
                        parsed = {}

                    if type(parsed.get("criteria_met")) is not bool:
                        parsed = self.fallback_parse(raw)

                    if type(parsed.get("criteria_met")) is bool:
                        parsed.setdefault(
                            "explanation",
                            "No explanation provided",
                        )
                        parsed["raw_judge_output"] = raw
                        results[idx] = parsed
                    else:
                        pending.append(idx)

            attempt += 1
            if pending and attempt > self.max_retries:
                raise RuntimeError(
                    "HealthBench judge failed to return valid criteria_met JSON "
                    f"after {attempt} attempts. First failed indices: {pending[:10]}"
                )

        return [x for x in results if x is not None]

    def close(self):
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass
        release_cuda()


def build_grader_prompt(
    hb,
    prompt_messages,
    response_text: str,
    rubric_item,
) -> str:
    """
    Exact causal-evaluator policy:
      original HealthBench conversation
      + generated assistant response
      + one HealthBench rubric item
      -> GRADER_TEMPLATE
    """
    convo = list(prompt_messages) + [
        {"content": response_text, "role": "assistant"}
    ]
    convo_str = "\n\n".join(
        f"{m['role']}: {m['content']}"
        for m in convo
    )
    return (
        hb.GRADER_TEMPLATE
        .replace("<<conversation>>", convo_str)
        .replace("<<rubric_item>>", str(rubric_item))
    )


def compute_healthbench_metrics(
    hb,
    example_tags,
    rubric_items,
    grades,
) -> dict[str, float]:
    overall = hb.calculate_score(rubric_items, grades)
    if overall is None:
        raise RuntimeError("HealthBench overall score unexpectedly returned None")

    metrics: dict[str, float] = {
        "overall_score": float(overall)
    }

    # Keep all simple-evals tags, exactly as in the causal evaluator.
    for tag in example_tags or []:
        metrics[str(tag)] = float(overall)

    tag_map: dict[str, list[tuple[Any, dict[str, Any]]]] = {}
    for rubric_item, grade in zip(rubric_items, grades, strict=True):
        for tag in rubric_item.tags:
            tag_map.setdefault(str(tag), []).append(
                (rubric_item, grade)
            )

    for tag, pairs in tag_map.items():
        items = [x[0] for x in pairs]
        item_grades = [x[1] for x in pairs]
        score = hb.calculate_score(items, item_grades)
        if score is not None:
            metrics[tag] = float(score)

    return metrics


def evaluation_cache_key(
    row: dict[str, Any],
    judge_model: str,
):
    return (
        str(row["prompt_id"]),
        str(row["model"]),
        str(row["condition"]),
        sha256_text(str(row["response_text"])),
        str(judge_model),
    )


def clipped_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.clip(np.mean(values), 0.0, 1.0))


def bootstrap_mean_ci(values, n_boot: int, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan

    point = clipped_mean(values)
    n = len(values)
    boots = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = clipped_mean(values[idx])

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_bootstrap_delta(a, b, n_boot: int, rng):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]

    if len(a) == 0:
        return np.nan, np.nan, np.nan

    point = clipped_mean(a) - clipped_mean(b)
    n = len(a)
    boots = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = (
            clipped_mean(a[idx])
            - clipped_mean(b[idx])
        )

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def make_summary(
    scores: pd.DataFrame,
    out_path: Path,
    n_boot: int,
    seed: int,
):
    """
    Same reporting structure as the causal evaluator:
      * all
      * insufficient
      * sufficient
    for HealthBench overall + axis metrics.
    """
    rng = np.random.default_rng(seed)
    rows = []

    groups = [
        ("all", scores),
        (
            "insufficient",
            scores[scores["evidence_label"] == 1],
        ),
        (
            "sufficient",
            scores[scores["evidence_label"] == 0],
        ),
    ]

    for label_group, subset in groups:
        for (model, condition), g in subset.groupby(
            ["model", "condition"]
        ):
            for metric in REPORT_METRICS:
                if metric not in g.columns:
                    continue

                vals = pd.to_numeric(
                    g[metric],
                    errors="coerce",
                ).to_numpy()
                vals = vals[np.isfinite(vals)]

                if not len(vals):
                    continue

                mean, lo, hi = bootstrap_mean_ci(
                    vals,
                    n_boot,
                    rng,
                )

                rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "label_group": label_group,
                        "metric": metric,
                        "n": int(len(vals)),
                        "mean": mean,
                        "ci95_low": lo,
                        "ci95_high": hi,
                    }
                )

    pd.DataFrame(rows).to_csv(
        out_path,
        index=False,
    )


def make_paired(
    scores: pd.DataFrame,
    out_path: Path,
    baseline_condition: str,
    n_boot: int,
    seed: int,
):
    """
    Paired intervention-vs-no_intervention comparison, identical in principle
    to the causal experiment evaluation.
    """
    rng = np.random.default_rng(seed + 1)
    rows = []

    groups = [
        ("all", scores),
        (
            "insufficient",
            scores[scores["evidence_label"] == 1],
        ),
        (
            "sufficient",
            scores[scores["evidence_label"] == 0],
        ),
    ]

    for label_group, subset in groups:
        for model, model_df in subset.groupby("model"):
            conditions = sorted(
                model_df["condition"].dropna().unique()
            )

            if baseline_condition not in conditions:
                continue

            base = (
                model_df[
                    model_df["condition"]
                    == baseline_condition
                ]
                .set_index("prompt_id")
            )

            for condition in conditions:
                if condition == baseline_condition:
                    continue

                cand = (
                    model_df[
                        model_df["condition"]
                        == condition
                    ]
                    .set_index("prompt_id")
                )

                common = base.index.intersection(
                    cand.index
                )
                if not len(common):
                    continue

                for metric in REPORT_METRICS:
                    if (
                        metric not in base.columns
                        or metric not in cand.columns
                    ):
                        continue

                    pair = pd.DataFrame(
                        {
                            "candidate": pd.to_numeric(
                                cand.loc[common, metric],
                                errors="coerce",
                            ),
                            "baseline": pd.to_numeric(
                                base.loc[common, metric],
                                errors="coerce",
                            ),
                        }
                    ).dropna()

                    if not len(pair):
                        continue

                    delta, lo, hi = paired_bootstrap_delta(
                        pair["candidate"].to_numpy(),
                        pair["baseline"].to_numpy(),
                        n_boot,
                        rng,
                    )

                    rows.append(
                        {
                            "model": model,
                            "condition": condition,
                            "baseline_condition": baseline_condition,
                            "label_group": label_group,
                            "metric": metric,
                            "n_paired": int(len(pair)),
                            "delta_candidate_minus_baseline": delta,
                            "delta_ci95_low": lo,
                            "delta_ci95_high": hi,
                        }
                    )

    pd.DataFrame(rows).to_csv(
        out_path,
        index=False,
    )


def load_healthbench_examples(
    hb,
    path: Path,
):
    """
    Load the rubric-capable HealthBench Hard file and build several independent
    lookup routes.

    Important: BRIDGE manifest IDs such as hard_000012_e90954de are NOT the
    original HealthBench prompt_id values. Therefore evaluation must not depend
    only on prompt_id equality.
    """
    rows: list[dict[str, Any]] = []
    id_map: dict[str, int] = {}
    fingerprint_map: dict[str, list[int]] = {}

    for source_index, raw in enumerate(
        read_jsonl(path)
    ):
        ex = dict(raw)

        required = {
            "prompt",
            "rubrics",
        }

        missing = required - set(ex)

        if missing:
            raise KeyError(
                f"HealthBench source row {source_index} missing "
                f"{sorted(missing)}"
            )

        ex["rubrics"] = [
            hb.RubricItem.from_dict(r)
            for r in ex["rubrics"]
        ]

        ex["__source_index"] = int(
            source_index
        )

        rows.append(
            ex
        )

        for field in (
            "prompt_id",
            "example_id",
            "id",
        ):
            if ex.get(field) is not None:
                key = str(
                    ex[field]
                )

                if key in id_map:
                    raise RuntimeError(
                        f"Duplicate HealthBench ID {key!r}"
                    )

                id_map[
                    key
                ] = source_index

        fp = messages_fingerprint(
            prompt_to_messages(
                ex["prompt"]
            )
        )

        fingerprint_map.setdefault(
            fp,
            [],
        ).append(
            source_index
        )

    if not rows:
        raise ValueError(
            f"HealthBench file is empty: {path}"
        )

    print()
    print("=" * 80)
    print("HEALTHBENCH RUBRIC SOURCE")
    print("=" * 80)
    print("file:", path)
    print("rows:", len(rows))
    print("direct IDs:", len(id_map))
    print(
        "unique prompt fingerprints:",
        sum(
            len(v) == 1
            for v in fingerprint_map.values()
        ),
    )
    print("=" * 80)

    return {
        "rows": rows,
        "id_map": id_map,
        "fingerprint_map": fingerprint_map,
        "path": Path(path),
    }


def _manifest_lookup(
    manifest: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    lookup = {}

    for _, row in manifest.iterrows():
        exid = str(
            row["example_id"]
        )

        if exid in lookup:
            raise RuntimeError(
                f"Duplicate manifest example_id {exid}"
            )

        lookup[
            exid
        ] = row.to_dict()

    return lookup


def resolve_healthbench_example(
    response_row: dict[str, Any],
    examples,
    manifest_lookup: dict[str, dict[str, Any]],
):
    """
    Resolve an M2 generation to the original HealthBench rubric row.

    Resolution order:
      1. direct raw HealthBench ID, if available;
      2. canonical prompt fingerprint from the immutable manifest;
      3. source_line audit (supports one-based and zero-based conventions).

    This directly fixes the prior KeyError where BRIDGE manifest IDs such as
    'hard_000012_e90954de' were incorrectly assumed to be raw HealthBench IDs.
    """
    rows = examples["rows"]
    id_map = examples["id_map"]
    fingerprint_map = examples["fingerprint_map"]

    # ------------------------------------------------------------------
    # 1. Direct raw ID match
    # ------------------------------------------------------------------

    for field in (
        "prompt_id",
        "example_id",
    ):
        value = response_row.get(
            field
        )

        if value is not None:
            key = str(
                value
            )

            if key in id_map:
                idx = id_map[
                    key
                ]

                return rows[
                    idx
                ], idx, "direct_id"

    # ------------------------------------------------------------------
    # Recover the corresponding immutable-manifest row.
    # ------------------------------------------------------------------

    manifest_id = str(
        response_row.get(
            "example_id",
            response_row.get(
                "prompt_id",
                "",
            ),
        )
    )

    if manifest_id not in manifest_lookup:
        raise KeyError(
            "M2 response ID is not present in immutable manifest: "
            f"{manifest_id!r}"
        )

    mrow = manifest_lookup[
        manifest_id
    ]

    target_messages = prompt_to_messages(
        mrow["prompt"]
    )

    target_fp = messages_fingerprint(
        target_messages
    )

    # ------------------------------------------------------------------
    # 2. Canonical conversation fingerprint
    # ------------------------------------------------------------------

    matches = fingerprint_map.get(
        target_fp,
        [],
    )

    if len(matches) == 1:
        idx = matches[
            0
        ]

        return rows[
            idx
        ], idx, "prompt_fingerprint"

    # ------------------------------------------------------------------
    # 3. source_line fallback, with prompt verification
    # ------------------------------------------------------------------

    source_line = response_row.get(
        "source_line"
    )

    if source_line is None:
        source_line = mrow.get(
            "source_line"
        )

    if source_line is not None and not pd.isna(source_line):
        line = int(
            source_line
        )

        candidates = []

        # Manifest has historically used one-based source lines, but check both
        # conventions instead of assuming.
        for convention, idx in (
            (
                "source_line_one_based",
                line - 1,
            ),
            (
                "source_line_zero_based",
                line,
            ),
        ):
            if 0 <= idx < len(rows):
                candidate_fp = messages_fingerprint(
                    prompt_to_messages(
                        rows[idx]["prompt"]
                    )
                )

                if candidate_fp == target_fp:
                    candidates.append(
                        (
                            convention,
                            idx,
                        )
                    )

        if len(candidates) == 1:
            convention, idx = candidates[
                0
            ]

            return rows[
                idx
            ], idx, convention

        if len(candidates) > 1:
            # This should only occur for duplicate adjacent prompts.
            convention, idx = candidates[
                0
            ]

            return rows[
                idx
            ], idx, convention + "_ambiguous_but_prompt_matched"

    raise KeyError(
        "Could not map M2 generation to HealthBench rubric row. "
        f"manifest_id={manifest_id!r}, "
        f"source_line={source_line!r}, "
        f"fingerprint_matches={matches}. "
        "This means the selected HealthBench rubric JSONL does not correspond "
        "to the immutable manifest used to generate M2."
    )


def run_causal_policy_evaluation(
    args,
    generation_path: Path,
):
    """
    Evaluate M2 exactly with the causal-experiment HealthBench rubric policy.

    Outputs are placed in:
        <output-dir>/causal_policy_eval/

    Output names intentionally match the causal evaluator:
        rubric_grades.jsonl
        response_scores.csv
        summary.csv
        paired_vs_baseline.csv
        metadata.json
    """
    if args.healthbench_jsonl is None:
        raise ValueError(
            "--healthbench-jsonl is required for --stage evaluate/all"
        )
    if args.simple_evals_repo is None:
        raise ValueError(
            "--simple-evals-repo is required for --stage evaluate/all"
        )
    if not args.healthbench_jsonl.exists():
        raise FileNotFoundError(args.healthbench_jsonl)
    if not args.simple_evals_repo.exists():
        raise FileNotFoundError(args.simple_evals_repo)
    if not generation_path.exists():
        raise FileNotFoundError(generation_path)

    outdir = args.output_dir / "causal_policy_eval"
    outdir.mkdir(parents=True, exist_ok=True)

    grades_path = outdir / "rubric_grades.jsonl"
    scores_path = outdir / "response_scores.csv"
    summary_path = outdir / "summary.csv"
    paired_path = outdir / "paired_vs_baseline.csv"
    metadata_path = outdir / "metadata.json"

    if args.force_evaluate:
        for path in (
            grades_path,
            scores_path,
            summary_path,
            paired_path,
            metadata_path,
        ):
            if path.exists():
                path.unlink()

    hb, hb_file = load_healthbench_reference(
        args.simple_evals_repo
    )
    examples = load_healthbench_examples(
        hb,
        args.healthbench_jsonl,
    )

    manifest_df = load_manifest(
        args.manifest
    )
    manifest_by_id = _manifest_lookup(
        manifest_df
    )

    responses = read_jsonl(generation_path)
    required_response_fields = {
        "prompt_id",
        "model",
        "condition",
        "response_text",
        "evidence_label",
    }

    for i, row in enumerate(responses):
        missing = required_response_fields - set(row)
        if missing:
            raise ValueError(
                f"M2 response row {i} missing fields: {sorted(missing)}"
            )
        # Fail early if the rubric source does not match the generated data.
        resolve_healthbench_example(
            row,
            examples,
            manifest_by_id,
        )

    existing = read_jsonl(grades_path) if grades_path.exists() else []
    existing_keys = {
        (
            str(r["prompt_id"]),
            str(r["model"]),
            str(r["condition"]),
            str(r["response_sha256"]),
            str(r.get("judge_model")),
        )
        for r in existing
    }

    pending = [
        r
        for r in responses
        if (
            evaluation_cache_key(r, args.judge_model)
            not in existing_keys
        )
    ]

    print()
    print("=" * 80)
    print("M2 - CAUSAL-POLICY HEALTHBENCH EVALUATION")
    print("=" * 80)
    print("HealthBench:", args.healthbench_jsonl)
    print("simple-evals:", hb_file)
    print("Responses:", generation_path)
    print("Judge:", args.judge_model)
    print("Cached response grades:", len(existing))
    print("Pending response grades:", len(pending))
    print("Policy: exact GRADER_TEMPLATE + calculate_score")
    print("Behavior classifier: NONE")
    print("=" * 80)

    judge = None
    if pending:
        judge = LocalHealthBenchJudge(
            model_name=args.judge_model,
            batch_size=args.judge_batch_size,
            max_new_tokens=args.judge_max_new_tokens,
            max_retries=args.max_json_retries,
        )

    try:
        for i, row in enumerate(pending, start=1):
            ex, hb_source_index, hb_resolution_method = resolve_healthbench_example(
                row,
                examples,
                manifest_by_id,
            )
            rubrics = ex["rubrics"]

            grader_prompts = [
                build_grader_prompt(
                    hb,
                    ex["prompt"],
                    str(row["response_text"]),
                    rubric,
                )
                for rubric in rubrics
            ]

            print(
                f"[{i}/{len(pending)}] "
                f"{row['model']} | {row['condition']} | "
                f"{row['prompt_id']} | {len(rubrics)} rubrics",
                flush=True,
            )

            start = time.time()
            grades = judge.grade(
                grader_prompts,
                hb.parse_json_to_dict,
            )
            metrics = compute_healthbench_metrics(
                hb,
                ex.get("example_tags", []),
                rubrics,
                grades,
            )

            rubric_items = []
            for rubric, grade in zip(
                rubrics,
                grades,
                strict=True,
            ):
                rubric_items.append(
                    {
                        **rubric.to_dict(),
                        "criteria_met": grade["criteria_met"],
                        "explanation": grade.get(
                            "explanation",
                            "No explanation provided",
                        ),
                        "raw_judge_output": grade.get(
                            "raw_judge_output"
                        ),
                    }
                )

            result = {
                "prompt_id": str(row["prompt_id"]),
                "model": row["model"],
                "condition": row["condition"],
                "evidence_label": int(row["evidence_label"]),
                "response_text": row["response_text"],
                "response_sha256": sha256_text(
                    str(row["response_text"])
                ),
                "judge_model": args.judge_model,
                "judge_type": "local_llm_healthbench_rubric",
                "grading_seconds": time.time() - start,
                "healthbench_source_index": int(hb_source_index),
                "healthbench_resolution_method": hb_resolution_method,
                "metrics": metrics,
                "rubric_items": rubric_items,
            }

            for field in M2_EVAL_FIELDS:
                if field not in result:
                    result[field] = row.get(field)

            append_jsonl(
                grades_path,
                result,
            )

    finally:
        if judge is not None:
            judge.close()

    all_grades = read_jsonl(grades_path)

    # Keep only the current judge in case a user deliberately reused a directory.
    all_grades = [
        r
        for r in all_grades
        if str(r.get("judge_model")) == str(args.judge_model)
    ]

    flat_rows = []
    for r in all_grades:
        flat = {
            "prompt_id": r["prompt_id"],
            "model": r["model"],
            "condition": r["condition"],
            "evidence_label": r.get("evidence_label"),
            "response_sha256": r["response_sha256"],
            "judge_model": r.get("judge_model"),
            "judge_type": r.get("judge_type"),
            "grading_seconds": r.get("grading_seconds"),
        }

        for field in M2_EVAL_FIELDS:
            if field not in flat:
                flat[field] = r.get(field)

        for name, value in r["metrics"].items():
            flat[name] = value

        flat_rows.append(flat)

    scores = pd.DataFrame(flat_rows)
    if not scores.empty:
        scores = scores.drop_duplicates(
            subset=[
                "prompt_id",
                "model",
                "condition",
                "response_sha256",
                "judge_model",
            ],
            keep="last",
        )

    scores.to_csv(
        scores_path,
        index=False,
    )

    make_summary(
        scores,
        summary_path,
        args.bootstrap,
        args.bootstrap_seed,
    )

    make_paired(
        scores,
        paired_path,
        baseline_condition="no_intervention",
        n_boot=args.bootstrap,
        seed=args.bootstrap_seed,
    )

    expected_responses = len(responses)
    if len(scores) != expected_responses:
        raise RuntimeError(
            "Evaluation row-count mismatch: "
            f"expected {expected_responses}, found {len(scores)}"
        )

    metadata = {
        "experiment": "M2",
        "evaluation_policy": (
            "same_as_causal_experiments_healthbench_rubric"
        ),
        "healthbench_jsonl": str(
            args.healthbench_jsonl.resolve()
        ),
        "responses_jsonl": str(
            generation_path.resolve()
        ),
        "simple_evals_healthbench_eval_py": str(
            hb_file.resolve()
        ),
        "judge_model": args.judge_model,
        "judge_type": "local_llm",
        "healthbench_template_source": (
            "OpenAI simple-evals GRADER_TEMPLATE"
        ),
        "healthbench_score_source": (
            "OpenAI simple-evals calculate_score"
        ),
        "baseline_condition": "no_intervention",
        "report_metrics": REPORT_METRICS,
        "stratification": [
            "all",
            "insufficient",
            "sufficient",
        ],
        "comparison": (
            "paired candidate-minus-baseline bootstrap"
        ),
        "bootstrap": int(args.bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "ask_abstain_direct_used": False,
        "regex_behavior_classifier_used": False,
        "healthbench_row_resolution": (
            "direct ID -> canonical prompt fingerprint -> verified source_line"
        ),
        "n_responses": int(len(scores)),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("M2 CAUSAL-POLICY EVALUATION COMPLETE")
    print("=" * 80)
    print("Rubric grades:", grades_path)
    print("Response scores:", scores_path)
    print("Summary:", summary_path)
    print("Paired comparison:", paired_path)
    print("Metadata:", metadata_path)
    print("=" * 80)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.rank != 4:
        print(
            "[WARNING] Final pre-specified M2 configuration was rank=4."
        )
    if args.alpha != 1.0:
        print(
            "[WARNING] Final pre-specified M2 configuration was alpha=1."
        )
    if args.behavior_train_limit != 0:
        print(
            "[WARNING] Final M2 configuration uses all TRAIN examples "
            "(--behavior-train-limit 0)."
        )
    if args.gate_floor != 1.0:
        raise ValueError(
            "Corrected final M2 requires --gate-floor 1."
        )

    for path in (
        args.manifest,
        args.probe_root,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    if args.stage in {"generate", "all"}:
        if not args.model_path.exists():
            raise FileNotFoundError(args.model_path)

    manifest = load_manifest(args.manifest)

    generation_path = (
        args.output_dir
        / "m2_generations.jsonl"
    )

    if args.stage in {"generate", "all"}:
        generation_path = generate_m2(
            args,
            manifest,
        )

    if args.stage in {"evaluate", "all"}:
        # Generation and judging are deliberately sequential; Qwen14 is loaded
        # only after the target model has been released by generate_m2().
        release_cuda()
        run_causal_policy_evaluation(
            args,
            generation_path,
        )

    run_metadata = {
        "experiment": "M2_FINAL_CAUSAL_EVAL_POLICY_V6",
        "model_key": args.model_key,
        "stage": args.stage,
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "behavior_train_limit": int(
            args.behavior_train_limit
        ),
        "gate_floor": float(args.gate_floor),
        "judge_model": args.judge_model,
        "evaluation_policy": (
            "HealthBench exact rubric evaluation; "
            "same policy as causal experiments"
        ),
        "ask_abstain_direct_used": False,
        "regex_behavior_classifier_used": False,
    }

    (
        args.output_dir
        / "run_metadata_causal_policy.json"
    ).write_text(
        json.dumps(
            run_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("BRIDGE M2 COMPLETE")
    print("=" * 80)
    print("Stage:", args.stage)
    print("Generation file:", generation_path)
    if args.stage in {"evaluate", "all"}:
        print(
            "Evaluation directory:",
            args.output_dir
            / "causal_policy_eval",
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
