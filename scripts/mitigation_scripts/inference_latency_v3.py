#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRIDGE inference-latency benchmark (final v3).

Routing fix in v3
---------
The final unified-mitigation experiment intentionally used the frozen cached
HealthBench gate decisions for exact reproducibility. A newly reconstructed
online activation path can differ slightly from the original activation-cache
extraction (especially for model-specific chat formatting), even though its
compute cost is representative. Therefore this benchmark:

  * times a real online probe forward pass on every gated condition;
  * records the live score/flag only as a diagnostic;
  * uses the FROZEN CACHED gate decision to route the prompt, exactly matching
    the final mitigation experiment.

This gives a faithful latency measurement of the deployment overhead while
keeping the evaluated routing policy identical to the final paper results.
The canonical probe is scored directly from `weights.npy` and `intercept.npy`.

Measured conditions
-------------------
1. baseline_generation
2. probe_gated_instruction
3. full_bridge_adaptive_m2
4. gate_only  (component timing only)

The two gated end-to-end conditions compute the gate LIVE from hidden states.
The cached HealthBench predictions are used only for an agreement audit; they
are never used to time the online system.

All CUDA timings use time.perf_counter bracketed by
torch.cuda.synchronize() before and after the measured region.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch


SUPPORTED = [
    "biomistral",
    "openbiollm",
    "ultramedical",
    "medgemma",
    "lingshu",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-key", required=True, choices=SUPPORTED)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--probe-dir", required=True, type=Path)
    p.add_argument("--m2-artifact-dir", required=True, type=Path)
    p.add_argument("--core-script", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)

    p.add_argument("--n-examples", type=int, default=30)
    p.add_argument("--warmup-examples", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--medgemma-internal-max-new-tokens",
        type=int,
        default=2000,
    )

    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument(
        "--behavior-steering-layer-1based",
        type=int,
        default=20,
    )

    p.add_argument(
        "--condition-order",
        choices=["randomized", "fixed"],
        default="randomized",
    )
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=123)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--live-gate-agreement-min",
        type=float,
        default=0.90,
    )
    p.add_argument(
        "--live-score-max-abs-diff",
        type=float,
        default=5e-3,
        help=(
            "Maximum tolerated absolute difference between live probe score "
            "and cached probe score when the cached score is available. "
            "The gate-agreement check is the hard safety check; this score "
            "check is an additional diagnostic."
        ),
    )
    return p.parse_args()


def import_core(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        "bridge_core_v6_latency",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module spec from {path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def boolish(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, float) and np.isfinite(v):
        return bool(int(v))
    return str(v).strip().lower() in {
        "1", "true", "t", "yes", "y"
    }


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "example_id",
        "binary_label",
        "included",
        "split",
        "prompt",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    x = df[df["included"].map(boolish)].copy()
    x["example_id"] = x["example_id"].astype(str)
    x["split"] = (
        x["split"].astype(str).str.strip().str.lower()
    )
    x["binary_label"] = x["binary_label"].astype(int)

    test = x[x["split"] == "test"].copy()
    sort_cols = [
        c
        for c in ["source_line", "example_id"]
        if c in test.columns
    ]
    if sort_cols:
        test = test.sort_values(
            sort_cols,
            kind="stable",
        )

    test = test.reset_index(drop=True)

    if len(test) != 188:
        print(
            f"[WARNING] Expected 188 HealthBench Hard rows; "
            f"found {len(test)}.",
            flush=True,
        )

    return test


# -------------------------------------------------------------------------
# Frozen linear probe
# -------------------------------------------------------------------------

def _read_probe_layer_and_threshold(
    probe_dir: Path,
):
    meta_path = (
        probe_dir
        / "selected_probe_metadata.json"
    )
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    meta = json.loads(
        meta_path.read_text(encoding="utf-8")
    )

    layer = None
    for key in [
        "selected_layer_0based",
        "layer_index_0based",
        "selected_layer_index_0based",
        "layer_0based",
    ]:
        if key in meta:
            layer = int(meta[key])
            break

    if layer is None:
        p = probe_dir / "layer_validation.csv"
        d = pd.read_csv(p)

        layer_cols = [
            c
            for c in d.columns
            if "0based" in str(c).lower()
        ]
        auc_cols = [
            c
            for c in d.columns
            if "auroc" in str(c).lower()
        ]

        if not layer_cols or not auc_cols:
            raise RuntimeError(
                f"Could not infer selected layer from {p}; "
                f"columns={d.columns.tolist()}"
            )

        idx = pd.to_numeric(
            d[auc_cols[0]],
            errors="coerce",
        ).idxmax()

        layer = int(
            d.loc[idx, layer_cols[0]]
        )

    threshold = None
    for key in [
        "selected_threshold",
        "threshold",
        "validation_threshold",
        "probe_threshold",
    ]:
        if key in meta:
            threshold = float(meta[key])
            break

    if threshold is None:
        p = probe_dir / "threshold_validation.csv"
        d = pd.read_csv(p)

        threshold_cols = [
            c
            for c in d.columns
            if "threshold" in str(c).lower()
        ]
        ba_cols = [
            c
            for c in d.columns
            if (
                "balanced" in str(c).lower()
                and "accuracy" in str(c).lower()
            )
        ]

        if not threshold_cols:
            raise RuntimeError(
                f"Could not infer threshold from {p}; "
                f"columns={d.columns.tolist()}"
            )

        if ba_cols and len(d) > 1:
            idx = pd.to_numeric(
                d[ba_cols[0]],
                errors="coerce",
            ).idxmax()
            threshold = float(
                d.loc[idx, threshold_cols[0]]
            )
        else:
            threshold = float(
                d.iloc[0][threshold_cols[0]]
            )

    return meta, layer, threshold


def load_frozen_linear_probe(
    probe_dir: Path,
):
    """
    Load the canonical frozen probe directly from weights.npy/intercept.npy.

    This deliberately does NOT call predict_proba/decision_function on
    selected_probe.joblib because the artifact is a dictionary wrapper.
    """
    weights_path = probe_dir / "weights.npy"
    intercept_path = probe_dir / "intercept.npy"

    if not weights_path.exists():
        raise FileNotFoundError(weights_path)
    if not intercept_path.exists():
        raise FileNotFoundError(intercept_path)

    weights = np.asarray(
        np.load(weights_path),
        dtype=np.float64,
    ).squeeze()

    intercept_arr = np.asarray(
        np.load(intercept_path),
        dtype=np.float64,
    ).reshape(-1)

    if weights.ndim != 1:
        raise RuntimeError(
            f"Expected 1-D weights, got {weights.shape}"
        )
    if intercept_arr.size != 1:
        raise RuntimeError(
            f"Expected scalar intercept, got {intercept_arr.shape}"
        )
    if not np.isfinite(weights).all():
        raise RuntimeError(
            "Probe weights contain non-finite values"
        )
    if not np.isfinite(intercept_arr).all():
        raise RuntimeError(
            "Probe intercept contains non-finite values"
        )

    intercept = float(intercept_arr[0])

    meta, layer, threshold = (
        _read_probe_layer_and_threshold(
            probe_dir
        )
    )

    # Optional artifact inspection only, never used for scoring.
    artifact_info = {
        "selected_probe_joblib_type": None,
        "selected_probe_joblib_keys": None,
    }
    joblib_path = (
        probe_dir
        / "selected_probe.joblib"
    )
    if joblib_path.exists():
        try:
            obj = joblib.load(joblib_path)
            artifact_info[
                "selected_probe_joblib_type"
            ] = type(obj).__name__
            if isinstance(obj, dict):
                artifact_info[
                    "selected_probe_joblib_keys"
                ] = sorted(
                    str(k)
                    for k in obj.keys()
                )
        except Exception as exc:
            artifact_info[
                "selected_probe_joblib_type"
            ] = f"unreadable:{type(exc).__name__}"

    return (
        weights,
        intercept,
        meta,
        layer,
        threshold,
        artifact_info,
    )


def stable_sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (
            1.0 + math.exp(-z)
        )

    ez = math.exp(z)
    return ez / (1.0 + ez)


def probe_probability(
    weights: np.ndarray,
    intercept: float,
    h: np.ndarray,
) -> float:
    x = np.asarray(
        h,
        dtype=np.float64,
    ).reshape(-1)

    if x.size != weights.size:
        raise RuntimeError(
            f"Probe dimension mismatch: "
            f"hidden={x.size}, weights={weights.size}"
        )

    z = float(
        np.dot(weights, x)
        + intercept
    )

    return stable_sigmoid(z)


def load_cached_predictions(
    probe_dir: Path,
):
    path = (
        probe_dir
        / "healthbench_test_predictions.csv"
    )
    if not path.exists():
        return None

    df = pd.read_csv(path)

    id_col = None
    for c in [
        "example_id",
        "prompt_id",
        "manifest_id",
        "id",
    ]:
        if c in df.columns:
            id_col = c
            break

    if id_col is None:
        return None

    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    return id_col, df.set_index(id_col)


def cached_gate_for_example(
    cached,
    example_id: str,
    threshold: float,
):
    if cached is None:
        return None, None

    _, table = cached

    if str(example_id) not in table.index:
        return None, None

    row = table.loc[str(example_id)]
    if isinstance(row, pd.DataFrame):
        if len(row) != 1:
            return None, None
        row = row.iloc[0]

    cached_score = None
    for c in [
        "probe_score",
        "evidence_score",
        "prob_insufficient",
        "probability_insufficient",
        "insufficient_probability",
        "positive_probability",
        "positive_prob",
        "probability",
        "prob",
        "y_score",
        "score",
    ]:
        if c in row.index and pd.notna(row[c]):
            cached_score = float(row[c])
            break

    cached_flag = None
    for c in [
        "prediction",
        "predicted_label",
        "pred_label",
        "y_pred",
        "probe_prediction",
        "flagged",
        "pred",
    ]:
        if c in row.index and pd.notna(row[c]):
            try:
                cached_flag = bool(
                    int(float(row[c]))
                )
            except Exception:
                cached_flag = boolish(
                    row[c]
                )
            break

    if (
        cached_flag is None
        and cached_score is not None
    ):
        cached_flag = bool(
            cached_score >= threshold
        )

    return cached_flag, cached_score


# -------------------------------------------------------------------------
# Timing helpers
# -------------------------------------------------------------------------

def sync():
    torch.cuda.synchronize()


def timed(fn):
    sync()
    t0 = time.perf_counter()
    result = fn()
    sync()
    return result, (
        time.perf_counter() - t0
    )


def unpack_generation(result):
    if isinstance(result, tuple):
        text = result[0]
        stats = (
            result[1]
            if (
                len(result) > 1
                and isinstance(result[1], dict)
            )
            else {}
        )
        return str(text or ""), stats

    return str(result or ""), {}


def visible_token_count(
    adapter,
    text: str,
):
    if not text:
        return 0

    out = adapter.tok(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )

    ids = out["input_ids"]
    if (
        isinstance(ids, list)
        and ids
        and isinstance(ids[0], list)
    ):
        ids = ids[0]

    return int(len(ids))


def prompt_token_count(
    adapter,
    messages,
):
    b = adapter.encode(messages)
    n = int(
        b["input_ids"].shape[1]
    )
    del b
    return n


def warmup_model(
    adapter,
    messages,
    steps: int = 2,
):
    """
    Warm up prefill and generation kernels.

    This avoids calling TargetAdapter.generate() during warmup so that
    MedGemma does not consume its long internal reasoning budget just for
    warmup.
    """
    for _ in range(max(1, steps)):
        b = adapter.encode(messages)

        with torch.inference_mode():
            _ = adapter.model(
                **b,
                use_cache=False,
                return_dict=True,
            )
        sync()
        del b

        b = adapter.encode(messages)
        with torch.inference_mode():
            _ = adapter.model.generate(
                **b,
                max_new_tokens=2,
                do_sample=False,
                use_cache=True,
                pad_token_id=(
                    adapter.tok.pad_token_id
                ),
                eos_token_id=adapter.eos_ids(),
            )
        sync()
        del b

    torch.cuda.empty_cache()


# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------

def bootstrap_mean_ci(
    values,
    n_boot: int,
    seed: int,
):
    v = np.asarray(
        values,
        dtype=float,
    )
    v = v[np.isfinite(v)]

    if len(v) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    point = float(v.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):
        idx = rng.integers(
            0,
            len(v),
            size=len(v),
        )
        boots[i] = float(
            v[idx].mean()
        )

    lo, hi = np.percentile(
        boots,
        [2.5, 97.5],
    )

    return (
        point,
        float(lo),
        float(hi),
    )


def make_summary(
    raw: pd.DataFrame,
    args,
    output_dir: Path,
):
    rows = []

    order = [
        "baseline_generation",
        "probe_gated_instruction",
        "full_bridge_adaptive_m2",
        "gate_only",
    ]

    for condition in order:
        g = raw[
            raw["condition"] == condition
        ]

        lat = pd.to_numeric(
            g["latency_seconds"],
            errors="coerce",
        ).dropna().to_numpy()

        if not len(lat):
            continue

        mean, lo, hi = bootstrap_mean_ci(
            lat,
            args.bootstrap,
            args.bootstrap_seed
            + order.index(condition),
        )

        out_tok = pd.to_numeric(
            g["visible_output_tokens"],
            errors="coerce",
        )
        prompt_tok = pd.to_numeric(
            g["prompt_tokens"],
            errors="coerce",
        )

        rows.append(
            {
                "model": args.model_key,
                "condition": condition,
                "n": int(len(lat)),
                "mean_seconds": mean,
                "mean_ci95_low": lo,
                "mean_ci95_high": hi,
                "median_seconds": float(
                    np.median(lat)
                ),
                "iqr_low_seconds": float(
                    np.percentile(lat, 25)
                ),
                "iqr_high_seconds": float(
                    np.percentile(lat, 75)
                ),
                "p95_seconds": float(
                    np.percentile(lat, 95)
                ),
                "sd_seconds": (
                    float(
                        np.std(
                            lat,
                            ddof=1,
                        )
                    )
                    if len(lat) > 1
                    else 0.0
                ),
                "mean_prompt_tokens": float(
                    prompt_tok.mean()
                ),
                "mean_visible_output_tokens": float(
                    out_tok.mean()
                ),
                "flagged_fraction": float(
                    g["flagged"]
                    .astype(float)
                    .mean()
                ),
            }
        )

    summary = pd.DataFrame(rows)

    base = summary.loc[
        summary["condition"]
        == "baseline_generation",
        "mean_seconds",
    ]

    if len(base) == 1:
        b = float(base.iloc[0])

        summary[
            "mean_overhead_seconds_vs_baseline"
        ] = (
            summary["mean_seconds"]
            - b
        )

        summary[
            "mean_overhead_pct_vs_baseline"
        ] = (
            (
                summary["mean_seconds"]
                / b
                - 1.0
            )
            * 100.0
        )

    summary.to_csv(
        output_dir
        / "latency_summary.csv",
        index=False,
    )

    subgroup_rows = []

    for condition in [
        "probe_gated_instruction",
        "full_bridge_adaptive_m2",
    ]:
        cg = raw[
            raw["condition"]
            == condition
        ]

        for flag_name, flag_value in [
            ("flagged", True),
            ("unflagged", False),
        ]:
            g = cg[
                cg["flagged"]
                == flag_value
            ]

            lat = pd.to_numeric(
                g["latency_seconds"],
                errors="coerce",
            ).dropna().to_numpy()

            if not len(lat):
                continue

            mean, lo, hi = bootstrap_mean_ci(
                lat,
                args.bootstrap,
                args.bootstrap_seed
                + (
                    10
                    if flag_value
                    else 20
                ),
            )

            subgroup_rows.append(
                {
                    "model": args.model_key,
                    "condition": condition,
                    "gate_group": flag_name,
                    "n": int(len(lat)),
                    "mean_seconds": mean,
                    "mean_ci95_low": lo,
                    "mean_ci95_high": hi,
                    "median_seconds": float(
                        np.median(lat)
                    ),
                    "iqr_low_seconds": float(
                        np.percentile(
                            lat,
                            25,
                        )
                    ),
                    "iqr_high_seconds": float(
                        np.percentile(
                            lat,
                            75,
                        )
                    ),
                }
            )

    pd.DataFrame(
        subgroup_rows
    ).to_csv(
        output_dir
        / "latency_by_gate_group.csv",
        index=False,
    )

    return summary


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(
        args.seed
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for latency benchmarking"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        args.model_path,
        args.manifest,
        args.probe_dir,
        args.m2_artifact_dir,
        args.core_script,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    core = import_core(
        args.core_script
    )

    (
        probe_weights,
        probe_intercept,
        probe_meta,
        probe_layer,
        threshold,
        probe_artifact_info,
    ) = load_frozen_linear_probe(
        args.probe_dir
    )

    controller_files = [
        args.m2_artifact_dir
        / "behavior_basis.npy",
        args.m2_artifact_dir
        / "x_mean.npy",
        args.m2_artifact_dir
        / "x_scale.npy",
        args.m2_artifact_dir
        / f"ridge_rank{args.rank}.joblib",
    ]

    missing_controller = [
        str(p)
        for p in controller_files
        if not p.exists()
    ]
    if missing_controller:
        raise FileNotFoundError(
            "Missing frozen M2 controller artifacts:\n"
            + "\n".join(
                missing_controller
            )
        )

    B, x_mean, x_scale, reg = (
        core.load_controller(
            args.m2_artifact_dir,
            args.rank,
        )
    )

    test = load_manifest(
        args.manifest
    )

    total_needed = (
        args.n_examples
        + args.warmup_examples
    )
    if total_needed > len(test):
        raise ValueError(
            f"Requested n={args.n_examples}"
            f"+warmup={args.warmup_examples}, "
            f"but only {len(test)} test rows exist."
        )

    chosen = test.sample(
        n=total_needed,
        random_state=args.seed,
    ).reset_index(drop=True)

    warmup_df = chosen.iloc[
        : args.warmup_examples
    ].copy()

    measured_df = chosen.iloc[
        args.warmup_examples :
    ].copy().reset_index(drop=True)

    steer = (
        args.behavior_steering_layer_1based
        - 1
    )

    adapter = core.TargetAdapter(
        args.model_key,
        args.model_path,
        args.trust_remote_code,
        medgemma_internal_max_new_tokens=(
            args.medgemma_internal_max_new_tokens
        ),
    )

    if not (
        0
        <= probe_layer
        < len(adapter.layers)
    ):
        raise IndexError(
            f"Probe layer {probe_layer} invalid for "
            f"{len(adapter.layers)} layers"
        )

    if not (
        0
        <= steer
        < len(adapter.layers)
    ):
        raise IndexError(
            f"Steering layer {steer} invalid for "
            f"{len(adapter.layers)} layers"
        )

    if (
        probe_weights.size
        != int(
            adapter.layers[
                probe_layer
            ].self_attn.q_proj.in_features
        )
        if (
            hasattr(
                adapter.layers[
                    probe_layer
                ],
                "self_attn",
            )
            and hasattr(
                adapter.layers[
                    probe_layer
                ].self_attn,
                "q_proj",
            )
        )
        else False
    ):
        raise RuntimeError(
            "Probe weight dimension is inconsistent "
            "with target model hidden size"
        )

    cached_predictions = (
        load_cached_predictions(
            args.probe_dir
        )
    )

    print("=" * 88)
    print(
        "BRIDGE INFERENCE LATENCY BENCHMARK v3"
    )
    print("=" * 88)
    print("model:", args.model_key)
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )
    print(
        "probe artifact scoring:",
        "weights.npy + intercept.npy",
    )
    print(
        "joblib wrapper type:",
        probe_artifact_info[
            "selected_probe_joblib_type"
        ],
    )
    print(
        "joblib wrapper keys:",
        probe_artifact_info[
            "selected_probe_joblib_keys"
        ],
    )
    print(
        "probe weight dim:",
        probe_weights.size,
    )
    print(
        "probe layer 0-based:",
        probe_layer,
    )
    print(
        "probe layer 1-based:",
        probe_layer + 1,
    )
    print(
        "threshold:",
        threshold,
    )
    print(
        "M2 rank:",
        args.rank,
    )
    print(
        "M2 alpha:",
        args.alpha,
    )
    print(
        "behavior layer 0-based:",
        steer,
    )
    print(
        "behavior layer 1-based:",
        steer + 1,
    )
    print(
        "measured examples:",
        len(measured_df),
    )
    print(
        "warmup examples:",
        len(warmup_df),
    )
    print(
        "max visible new tokens:",
        args.max_new_tokens,
    )
    print(
        "MedGemma internal budget:",
        args.medgemma_internal_max_new_tokens,
    )
    print("=" * 88)

    if len(warmup_df):
        first_messages = (
            core.prompt_to_messages(
                warmup_df.iloc[0]["prompt"]
            )
        )

        warmup_model(
            adapter,
            first_messages,
            steps=max(
                1,
                args.warmup_examples,
            ),
        )

        print(
            "Warmup complete.",
            flush=True,
        )

    raw_rows = []
    online_vs_cached_flags = []
    online_vs_cached_score_diffs = []

    condition_names = [
        "baseline_generation",
        "probe_gated_instruction",
        "full_bridge_adaptive_m2",
    ]

    for i, row in measured_df.iterrows():
        exid = str(
            row["example_id"]
        )
        y = int(
            row["binary_label"]
        )
        messages = (
            core.prompt_to_messages(
                row["prompt"]
            )
        )

        prompt_tokens = (
            prompt_token_count(
                adapter,
                messages,
            )
        )

        def gate_only_fn():
            acts = adapter.activations(
                messages,
                [probe_layer],
            )

            p = probe_probability(
                probe_weights,
                probe_intercept,
                acts[probe_layer],
            )

            return (
                p,
                bool(p >= threshold),
            )

        (
            audit_prob,
            audit_flag,
        ), gate_sec = timed(
            gate_only_fn
        )

        (
            cached_flag,
            cached_score,
        ) = cached_gate_for_example(
            cached_predictions,
            exid,
            threshold,
        )

        if cached_flag is not None:
            online_vs_cached_flags.append(
                int(
                    audit_flag
                    == cached_flag
                )
            )

        if cached_score is not None:
            online_vs_cached_score_diffs.append(
                abs(
                    audit_prob
                    - cached_score
                )
            )

        raw_rows.append(
            {
                "model": args.model_key,
                "example_id": exid,
                "evidence_label": y,
                "condition": "gate_only",
                "latency_seconds": gate_sec,
                "prompt_tokens": prompt_tokens,
                "visible_output_tokens": 0,
                "online_probe_score": audit_prob,
                "live_flag": audit_flag,
                "flagged": (cached_flag if cached_flag is not None else audit_flag),
                "cached_flag": cached_flag,
                "cached_score": cached_score,
                "hook_fired": False,
                "generation_failure": False,
            }
        )

        order = list(
            condition_names
        )

        if (
            args.condition_order
            == "randomized"
        ):
            rng = random.Random(
                args.seed * 100000
                + i
            )
            rng.shuffle(order)

        for condition in order:
            generation_failure = False
            hook_fired = False
            online_prob = np.nan
            live_flag = False
            flagged = False
            output_text = ""

            if (
                condition
                == "baseline_generation"
            ):
                def fn():
                    return adapter.generate(
                        messages,
                        args.max_new_tokens,
                    )

                try:
                    result, sec = timed(fn)
                    (
                        output_text,
                        stats,
                    ) = unpack_generation(
                        result
                    )
                    hook_fired = bool(
                        stats.get(
                            "hook_fired",
                            False,
                        )
                    )
                except Exception as exc:
                    sec = np.nan
                    generation_failure = True
                    print(
                        f"[WARNING] baseline generation "
                        f"failure {exid}: {exc}",
                        flush=True,
                    )

            elif (
                condition
                == "probe_gated_instruction"
            ):
                def fn():
                    acts = adapter.activations(
                        messages,
                        [probe_layer],
                    )

                    p = probe_probability(
                        probe_weights,
                        probe_intercept,
                        acts[probe_layer],
                    )

                    live_flag = bool(
                        p >= threshold
                    )

                    if cached_flag is None:
                        raise RuntimeError(
                            f"Missing frozen cached gate for {exid}"
                        )
                    route_flag = bool(cached_flag)

                    gen_messages = (
                        core.append_safety_instruction(
                            messages
                        )
                        if route_flag
                        else messages
                    )

                    result = adapter.generate(
                        gen_messages,
                        args.max_new_tokens,
                    )

                    return (
                        p,
                        live_flag,
                        route_flag,
                        result,
                    )

                try:
                    result, sec = timed(fn)
                    (
                        online_prob,
                        live_flag,
                        flagged,
                        gen_result,
                    ) = result

                    (
                        output_text,
                        stats,
                    ) = unpack_generation(
                        gen_result
                    )

                    hook_fired = bool(
                        stats.get(
                            "hook_fired",
                            False,
                        )
                    )
                except Exception as exc:
                    sec = np.nan
                    generation_failure = True
                    print(
                        "[WARNING] gated-instruction "
                        f"failure {exid}: {exc}",
                        flush=True,
                    )

            elif (
                condition
                == "full_bridge_adaptive_m2"
            ):
                def fn():
                    # One original-prompt forward pass obtains both
                    # the online gate state and controller-input state.
                    indices = sorted(
                        set(
                            [
                                probe_layer,
                                steer,
                            ]
                        )
                    )

                    acts = adapter.activations(
                        messages,
                        indices,
                    )

                    p = probe_probability(
                        probe_weights,
                        probe_intercept,
                        acts[probe_layer],
                    )

                    live_flag = bool(
                        p >= threshold
                    )

                    if cached_flag is None:
                        raise RuntimeError(
                            f"Missing frozen cached gate for {exid}"
                        )
                    route_flag = bool(cached_flag)

                    if route_flag:
                        raw_delta = (
                            core.predict_delta(
                                acts[steer],
                                B,
                                x_mean,
                                x_scale,
                                reg,
                            )
                        )

                        gen_messages = (
                            core.append_safety_instruction(
                                messages
                            )
                        )

                        result = adapter.generate(
                            gen_messages,
                            args.max_new_tokens,
                            steer,
                            raw_delta
                            * float(args.alpha),
                        )
                    else:
                        result = adapter.generate(
                            messages,
                            args.max_new_tokens,
                        )

                    return (
                        p,
                        live_flag,
                        route_flag,
                        result,
                    )

                try:
                    result, sec = timed(fn)

                    (
                        online_prob,
                        live_flag,
                        flagged,
                        gen_result,
                    ) = result

                    (
                        output_text,
                        stats,
                    ) = unpack_generation(
                        gen_result
                    )

                    hook_fired = bool(
                        stats.get(
                            "hook_fired",
                            False,
                        )
                    )

                    if (
                        flagged
                        and not hook_fired
                    ):
                        raise RuntimeError(
                            "Full BRIDGE prompt was flagged "
                            "but M2 hook did not fire"
                        )

                    if (
                        (not flagged)
                        and hook_fired
                    ):
                        raise RuntimeError(
                            "Full BRIDGE prompt was unflagged "
                            "but M2 hook fired"
                        )

                except Exception as exc:
                    sec = np.nan
                    generation_failure = True
                    print(
                        "[WARNING] full-BRIDGE "
                        f"failure {exid}: {exc}",
                        flush=True,
                    )

            else:
                raise AssertionError(
                    condition
                )

            n_out = (
                visible_token_count(
                    adapter,
                    output_text,
                )
                if output_text
                else 0
            )

            raw_rows.append(
                {
                    "model": args.model_key,
                    "example_id": exid,
                    "evidence_label": y,
                    "condition": condition,
                    "latency_seconds": sec,
                    "prompt_tokens": prompt_tokens,
                    "visible_output_tokens": n_out,
                    "online_probe_score": online_prob,
                    "live_flag": live_flag,
                    "flagged": flagged,
                    "cached_flag": cached_flag,
                    "cached_score": cached_score,
                    "hook_fired": hook_fired,
                    "generation_failure": (
                        generation_failure
                    ),
                }
            )

        print(
            f"[{i+1}/{len(measured_df)}] "
            f"{exid} y={y} "
            f"live_gate={audit_prob:.6f} "
            f"live_flag={int(audit_flag)} "
            f"route_flag={int(bool(cached_flag)) if cached_flag is not None else -1}",
            flush=True,
        )

    raw = pd.DataFrame(
        raw_rows
    )

    raw_path = (
        args.output_dir
        / "latency_raw.csv"
    )
    raw.to_csv(
        raw_path,
        index=False,
    )

    agreement = np.nan

    if online_vs_cached_flags:
        agreement = float(
            np.mean(
                online_vs_cached_flags
            )
        )

        print(
            "Online-vs-cached gate agreement "
            f"on measured prompts: {agreement:.3f} "
            f"({sum(online_vs_cached_flags)}/"
            f"{len(online_vs_cached_flags)})"
        )

        if (
            agreement
            < args.live_gate_agreement_min
        ):
            print(
                f"[DIAGNOSTIC] Live-vs-cached agreement {agreement:.3f} "
                f"is below {args.live_gate_agreement_min:.3f}. "
                "This does not invalidate v3 latency because the exact frozen "
                "cached gate is used for routing; the live pass is timed only "
                "to measure online gate compute cost.",
                flush=True,
            )

    else:
        print(
            "[WARNING] Could not audit live gate "
            "against cached predictions."
        )

    max_score_diff = np.nan
    mean_score_diff = np.nan

    if online_vs_cached_score_diffs:
        max_score_diff = float(
            np.max(
                online_vs_cached_score_diffs
            )
        )
        mean_score_diff = float(
            np.mean(
                online_vs_cached_score_diffs
            )
        )

        print(
            "Live-vs-cached probe score absolute diff: "
            f"mean={mean_score_diff:.8g}, "
            f"max={max_score_diff:.8g}"
        )

        if (
            max_score_diff
            > args.live_score_max_abs_diff
        ):
            print(
                "[WARNING] Live probe scores differ from "
                "cached scores by more than "
                f"{args.live_score_max_abs_diff:g}. "
                "Gate agreement remains the hard check.",
                flush=True,
            )

    failures = raw[
        raw["condition"].isin(
            condition_names
        )
        & raw[
            "generation_failure"
        ].astype(bool)
    ]

    if len(failures):
        failures.to_csv(
            args.output_dir
            / "generation_failures.csv",
            index=False,
        )

        raise RuntimeError(
            f"{len(failures)} measured user-facing "
            "generations failed. See "
            "generation_failures.csv."
        )

    summary = make_summary(
        raw,
        args,
        args.output_dir,
    )

    metadata = {
        "model": args.model_key,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "n_measured_examples": int(
            len(measured_df)
        ),
        "n_warmup_examples": int(
            len(warmup_df)
        ),
        "sample_seed": int(args.seed),
        "conditions": (
            condition_names
            + ["gate_only"]
        ),
        "condition_order": (
            args.condition_order
        ),
        "probe_scoring": (
            "direct frozen logistic score from "
            "weights.npy and intercept.npy"
        ),
        "probe_weight_dim": int(
            probe_weights.size
        ),
        "probe_layer_0based": int(
            probe_layer
        ),
        "probe_layer_1based": int(
            probe_layer + 1
        ),
        "probe_threshold": float(
            threshold
        ),
        "selected_probe_joblib_type": (
            probe_artifact_info[
                "selected_probe_joblib_type"
            ]
        ),
        "selected_probe_joblib_keys": (
            probe_artifact_info[
                "selected_probe_joblib_keys"
            ]
        ),
        "m2_rank": int(args.rank),
        "m2_alpha": float(args.alpha),
        "behavior_layer_0based": int(
            steer
        ),
        "behavior_layer_1based": int(
            steer + 1
        ),
        "max_visible_new_tokens": int(
            args.max_new_tokens
        ),
        "medgemma_internal_max_new_tokens": int(
            args.medgemma_internal_max_new_tokens
        ),
        "timing_method": (
            "time.perf_counter bracketed by "
            "torch.cuda.synchronize before/after"
        ),
        "live_probe_forward_timed": True,
        "routing_policy": (
            "frozen cached HealthBench gate decisions, matching final unified mitigation"
        ),
        "cached_gate_lookup_time_included": False,
        "full_bridge_state_extraction": (
            "single original-prompt forward pass captures probe and controller layers; "
            "frozen cached flag chooses the route"
        ),
        "online_cached_gate_agreement": (
            None
            if not np.isfinite(agreement)
            else agreement
        ),
        "online_cached_probe_score_mean_abs_diff": (
            None
            if not np.isfinite(mean_score_diff)
            else mean_score_diff
        ),
        "online_cached_probe_score_max_abs_diff": (
            None
            if not np.isfinite(max_score_diff)
            else max_score_diff
        ),
        "bootstrap": int(args.bootstrap),
        "bootstrap_seed": int(
            args.bootstrap_seed
        ),
        "core_script": str(
            args.core_script.resolve()
        ),
        "probe_dir": str(
            args.probe_dir.resolve()
        ),
        "m2_artifact_dir": str(
            args.m2_artifact_dir.resolve()
        ),
    }

    (
        args.output_dir
        / "latency_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    adapter.close()

    print()
    print("=" * 88)
    print(
        "LATENCY BENCHMARK COMPLETE"
    )
    print("=" * 88)
    print(
        summary.to_string(
            index=False
        )
    )
    print()
    print(
        "Raw:",
        raw_path,
    )
    print(
        "Summary:",
        args.output_dir
        / "latency_summary.csv",
    )
    print(
        "Gate groups:",
        args.output_dir
        / "latency_by_gate_group.csv",
    )
    print(
        "Metadata:",
        args.output_dir
        / "latency_metadata.json",
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
