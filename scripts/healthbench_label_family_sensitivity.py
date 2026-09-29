#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRIDGE HealthBench label-family sensitivity analysis.

Purpose
-------
Tests whether the evidence-insufficiency probe result depends on combining
HealthBench's two directly relevant physician-agreed label families:

1) Context-Seeking family
   sufficient   : enough-context
   insufficient : not-enough-context

2) Health-Data-Task family
   sufficient   : enough-info-to-complete-task
   insufficient : not-enough-info-to-complete-task

For each model and each label family, this script:
  * reuses the existing cached hidden activations (NO model loading, NO GPU);
  * preserves the immutable manifest's train/validation/test split;
  * trains one L2 logistic-regression probe per layer on TRAIN only;
  * selects the layer by VALIDATION AUROC;
  * refits the selected-layer probe on TRAIN only;
  * selects a decision threshold by VALIDATION balanced accuracy;
  * evaluates once on the held-out TEST split;
  * reports AUROC, AUPRC, balanced accuracy, sensitivity, specificity,
    F1 for the insufficient class, and stratified-bootstrap 95% CIs.

A "combined" analysis is also rerun as a sanity check. It should reproduce
the main BRIDGE linear-probe result closely. This is useful for verifying
that activation loading and the training protocol are correct before the
two sensitivity subsets are interpreted.

No test labels are used for fitting, layer selection, or threshold selection.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


SUPPORTED_MODELS = [
    "biomistral",
    "openbiollm",
    "ultramedical",
    "medgemma",
    "lingshu",
]

FAMILY_DEFINITIONS = {
    "context_seeking": {
        0: {"enough-context"},
        1: {"not-enough-context"},
    },
    "health_data_task": {
        0: {"enough-info-to-complete-task"},
        1: {"not-enough-info-to-complete-task"},
    },
}

EXPECTED_MAIN_LAYERS_0BASED = {
    "biomistral": 16,
    "openbiollm": 15,
    "ultramedical": 13,
    "medgemma": 28,
    "lingshu": 21,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path.home() / "JBHI" / "BRIDGE",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=SUPPORTED_MODELS,
        choices=SUPPORTED_MODELS,
    )
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=10000)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit labels/cache alignment only; do not train probes.",
    )
    p.add_argument(
        "--strict-main-layer-check",
        action="store_true",
        help=(
            "Fail if the rerun combined analysis does not select the same "
            "layer as the already-validated main BRIDGE probe."
        ),
    )
    return p.parse_args()


def boolish(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, float) and np.isfinite(v):
        return bool(int(v))
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def normalize_category(text: str) -> str:
    s = str(text).strip().lower()
    s = s.replace("_", "-")
    s = re.sub(r"\s+", "-", s)
    for prefix in [
        "physician-agreed-category:",
        "physician_agreed_category:",
        "physician-agreed-category-",
        "physician_agreed_category-",
    ]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    s = s.strip(" :-")
    return s


def parse_categories(value: Any) -> set[str]:
    """
    Robustly parse physician_agreed_categories from CSV cells that may contain
    JSON, Python-list repr, comma-separated text, or a single category.
    """
    if value is None:
        return set()

    if isinstance(value, float) and np.isnan(value):
        return set()

    if isinstance(value, (list, tuple, set)):
        raw = list(value)
    elif isinstance(value, dict):
        raw = list(value.keys())
    else:
        s = str(value).strip()
        if not s:
            return set()

        parsed = None
        if s.startswith("[") or s.startswith("{") or s.startswith("("):
            try:
                parsed = json.loads(s)
            except Exception:
                try:
                    parsed = ast.literal_eval(s)
                except Exception:
                    parsed = None

        if isinstance(parsed, dict):
            raw = list(parsed.keys())
        elif isinstance(parsed, (list, tuple, set)):
            raw = list(parsed)
        elif isinstance(parsed, str):
            raw = [parsed]
        else:
            # Pull explicit HealthBench category tokens first.
            hits = re.findall(
                r"(?:physician[_-]agreed[_-]category:)?"
                r"(?:not-enough-context|enough-context|"
                r"not-enough-info-to-complete-task|"
                r"enough-info-to-complete-task)",
                s.lower().replace("_", "-"),
            )
            if hits:
                raw = hits
            else:
                raw = re.split(r"[,;|]", s)

    out = set()
    for x in raw:
        c = normalize_category(x)
        if c:
            out.add(c)
    return out


def family_label(categories: set[str], family: str) -> int | None:
    defs = FAMILY_DEFINITIONS[family]
    matched = []

    for label, names in defs.items():
        if categories.intersection(names):
            matched.append(label)

    if len(matched) == 0:
        return None

    if len(set(matched)) != 1:
        raise RuntimeError(
            f"Conflicting {family} labels in categories={sorted(categories)}"
        )

    return int(matched[0])


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "example_id",
        "binary_label",
        "included",
        "split",
        "physician_agreed_categories",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Manifest missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df = df[df["included"].map(boolish)].copy()
    df["example_id"] = df["example_id"].astype(str)
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["binary_label"] = df["binary_label"].astype(int)
    df["__categories"] = df["physician_agreed_categories"].map(
        parse_categories
    )

    if df["example_id"].duplicated().any():
        dup = df.loc[df["example_id"].duplicated(), "example_id"].tolist()
        raise RuntimeError(
            f"Duplicate included example_id values, e.g. {dup[:5]}"
        )

    allowed_splits = {"train", "validation", "test"}
    bad_splits = sorted(set(df["split"]) - allowed_splits)
    if bad_splits:
        raise RuntimeError(f"Unexpected splits: {bad_splits}")

    return df.reset_index(drop=True)


def build_analysis_manifest(df: pd.DataFrame, analysis: str) -> pd.DataFrame:
    if analysis == "combined":
        out = df.copy()
        out["analysis_label"] = out["binary_label"].astype(int)
        return out

    labels = []
    for cats in df["__categories"]:
        labels.append(family_label(cats, analysis))

    out = df.copy()
    out["analysis_label"] = labels
    out = out[out["analysis_label"].notna()].copy()
    out["analysis_label"] = out["analysis_label"].astype(int)

    # Verify that the family-specific label agrees with the immutable main label.
    mismatch = out[
        out["analysis_label"].astype(int)
        != out["binary_label"].astype(int)
    ]
    if len(mismatch):
        cols = [
            "example_id",
            "binary_label",
            "analysis_label",
            "physician_agreed_categories",
        ]
        raise RuntimeError(
            f"{analysis}: {len(mismatch)} family labels disagree with the "
            f"immutable binary label.\n{mismatch[cols].head(10).to_string(index=False)}"
        )

    return out.reset_index(drop=True)


def print_subset_audit(df: pd.DataFrame):
    print()
    print("=" * 88)
    print("LABEL-FAMILY AUDIT")
    print("=" * 88)

    context_ids = set(
        build_analysis_manifest(df, "context_seeking")["example_id"]
    )
    health_ids = set(
        build_analysis_manifest(df, "health_data_task")["example_id"]
    )

    print("Included immutable-manifest examples:", len(df))
    print("Context-Seeking family:", len(context_ids))
    print("Health-Data-Task family:", len(health_ids))
    print("Family overlap:", len(context_ids & health_ids))
    print()

    for analysis in [
        "combined",
        "context_seeking",
        "health_data_task",
    ]:
        x = build_analysis_manifest(df, analysis)
        print(f"[{analysis}]")
        table = (
            x.groupby(["split", "analysis_label"])
            .size()
            .unstack(fill_value=0)
            .rename(columns={0: "sufficient_0", 1: "insufficient_1"})
        )
        print(table.to_string())
        print("Total:", len(x))
        print()

        for split in ["train", "validation", "test"]:
            sx = x[x["split"] == split]
            labels = set(sx["analysis_label"].astype(int))
            if labels != {0, 1}:
                raise RuntimeError(
                    f"{analysis}/{split} does not contain both classes: "
                    f"{sorted(labels)}"
                )

    print("=" * 88)


def extract_id_from_payload(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None

    for key in [
        "example_id",
        "prompt_id",
        "manifest_id",
        "id",
    ]:
        if key in payload and payload[key] is not None:
            return str(payload[key])

    meta = payload.get("metadata")
    if isinstance(meta, dict):
        for key in [
            "example_id",
            "prompt_id",
            "manifest_id",
            "id",
        ]:
            if key in meta and meta[key] is not None:
                return str(meta[key])

    return None


def find_activation_tensor(payload: Any) -> torch.Tensor:
    """
    Accept the common BRIDGE activation-cache layouts. The final tensor must
    represent [n_layers, hidden_dim] for one example.
    """
    if torch.is_tensor(payload):
        t = payload
    elif isinstance(payload, np.ndarray):
        t = torch.from_numpy(payload)
    elif isinstance(payload, dict):
        preferred = [
            "activations",
            "last_token_activations",
            "layer_activations",
            "hidden_states",
            "features",
            "activation",
        ]

        t = None
        for key in preferred:
            value = payload.get(key)
            if torch.is_tensor(value):
                t = value
                break
            if isinstance(value, np.ndarray):
                t = torch.from_numpy(value)
                break

        if t is None:
            candidates = []
            for key, value in payload.items():
                if torch.is_tensor(value):
                    candidates.append((key, value))
                elif isinstance(value, np.ndarray):
                    candidates.append((key, torch.from_numpy(value)))

            plausible = [
                (key, value)
                for key, value in candidates
                if value.ndim in {2, 3}
                and value.numel() > 1000
            ]

            if len(plausible) == 1:
                t = plausible[0][1]
            else:
                shapes = {
                    key: tuple(value.shape)
                    for key, value in candidates
                }
                raise RuntimeError(
                    "Could not uniquely identify activation tensor. "
                    f"Tensor fields={shapes}"
                )
    else:
        raise TypeError(
            f"Unsupported activation payload type: {type(payload)}"
        )

    t = t.detach().cpu()

    # Remove singleton batch/token dimensions if present.
    while t.ndim > 2 and t.shape[0] == 1:
        t = t.squeeze(0)

    if t.ndim == 3:
        # Common shape: [layers, 1, hidden].
        if t.shape[1] == 1:
            t = t[:, 0, :]
        # Common shape: [1, layers, hidden] handled above.
        else:
            raise RuntimeError(
                f"Ambiguous 3-D activation tensor shape: {tuple(t.shape)}"
            )

    if t.ndim != 2:
        raise RuntimeError(
            "Expected one-example activation matrix [layers, hidden], "
            f"got {tuple(t.shape)}"
        )

    if not torch.isfinite(t).all():
        raise RuntimeError("Activation tensor contains non-finite values")

    return t.float()


def load_pt(path: Path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def activation_id_from_file(path: Path, payload: Any) -> str:
    payload_id = extract_id_from_payload(payload)
    if payload_id:
        return payload_id

    # Extraction pipeline normally stores one file per immutable example.
    # Support "<example_id>.pt" and prefixes/suffixes containing the ID.
    return path.stem


def load_split_cache(
    activation_dir: Path,
    manifest_split: pd.DataFrame,
):
    files = sorted(activation_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(
            f"No .pt activation files in {activation_dir}"
        )

    wanted_ids = manifest_split["example_id"].astype(str).tolist()
    wanted_set = set(wanted_ids)

    by_id = {}
    unresolved = []

    for path in files:
        payload = load_pt(path)
        file_id = activation_id_from_file(path, payload)

        # Exact match first.
        if file_id in wanted_set:
            exid = file_id
        else:
            # Robust filename fallback: exactly one manifest ID occurs in stem.
            hits = [ex for ex in wanted_ids if ex in path.stem]
            if len(hits) == 1:
                exid = hits[0]
            else:
                unresolved.append(
                    (path.name, file_id, hits[:5])
                )
                continue

        if exid in by_id:
            raise RuntimeError(
                f"Duplicate activation for {exid}: {path}"
            )

        by_id[exid] = find_activation_tensor(payload).numpy()

    missing = [ex for ex in wanted_ids if ex not in by_id]

    if missing:
        preview = "\n".join(
            f"  file={a} parsed_id={b!r} candidate_hits={c}"
            for a, b, c in unresolved[:5]
        )
        raise RuntimeError(
            f"Activation/manifest alignment failed for {activation_dir}.\n"
            f"Manifest rows={len(wanted_ids)}, .pt files={len(files)}, "
            f"matched={len(by_id)}, missing={len(missing)}.\n"
            f"First missing IDs: {missing[:10]}\n"
            f"First unresolved files:\n{preview}"
        )

    mats = [by_id[ex] for ex in wanted_ids]
    shapes = {tuple(x.shape) for x in mats}
    if len(shapes) != 1:
        raise RuntimeError(
            f"Inconsistent activation shapes in {activation_dir}: {shapes}"
        )

    X = np.stack(mats, axis=0).astype(np.float32)
    return X


def load_model_cache(
    project_root: Path,
    model: str,
    manifest: pd.DataFrame,
):
    root = project_root / "cache" / "activations" / model / "healthbench"

    data = {}
    for split in ["train", "validation", "test"]:
        split_manifest = (
            manifest[manifest["split"] == split]
            .copy()
            .reset_index(drop=True)
        )

        activation_dir = root / split
        if not activation_dir.exists():
            raise FileNotFoundError(activation_dir)

        X = load_split_cache(
            activation_dir,
            split_manifest,
        )

        data[split] = {
            "manifest": split_manifest,
            "X": X,
        }

        print(
            f"Loaded {model}/healthbench/{split}: "
            f"X={X.shape}"
        )

    return data


def subset_cache_for_analysis(
    full_cache,
    analysis_manifest: pd.DataFrame,
):
    ids = set(
        analysis_manifest["example_id"].astype(str)
    )

    out = {}
    for split in ["train", "validation", "test"]:
        full_m = full_cache[split]["manifest"]
        full_X = full_cache[split]["X"]

        mask = full_m["example_id"].astype(str).isin(ids).to_numpy()
        m = full_m.loc[mask].copy().reset_index(drop=True)
        X = full_X[mask]

        # Pull family-specific analysis_label by example_id.
        label_map = analysis_manifest.set_index(
            "example_id"
        )["analysis_label"].astype(int).to_dict()

        y = np.asarray(
            [label_map[str(ex)] for ex in m["example_id"]],
            dtype=np.int64,
        )

        out[split] = {
            "manifest": m,
            "X": X,
            "y": y,
        }

        counts = np.bincount(y, minlength=2)
        print(
            f"  {split}: n={len(y)} "
            f"class0={int(counts[0])} class1={int(counts[1])}"
        )

    return out


def make_probe(C: float, seed: int, max_iter: int):
    return LogisticRegression(
        penalty="l2",
        C=C,
        class_weight="balanced",
        solver="liblinear",
        max_iter=max_iter,
        random_state=seed,
    )


def safe_auroc(y, scores):
    if len(np.unique(y)) != 2:
        return np.nan
    return float(roc_auc_score(y, scores))


def safe_auprc(y, scores):
    if len(np.unique(y)) != 2:
        return np.nan
    return float(average_precision_score(y, scores))


def choose_threshold(y_val, p_val):
    """
    Exhaustively evaluate thresholds at every distinct validation probability.
    This mirrors validation-only balanced-accuracy threshold selection.
    """
    thresholds = np.unique(
        np.concatenate([
            np.asarray([0.0]),
            np.asarray(p_val, dtype=float),
            np.asarray([1.0 + 1e-12]),
        ])
    )

    rows = []
    for threshold in thresholds:
        pred = (p_val >= threshold).astype(int)
        ba = balanced_accuracy_score(y_val, pred)
        sens = recall_score(
            y_val,
            pred,
            pos_label=1,
            zero_division=0,
        )
        spec = recall_score(
            y_val,
            pred,
            pos_label=0,
            zero_division=0,
        )
        rows.append({
            "threshold": float(threshold),
            "balanced_accuracy": float(ba),
            "sensitivity": float(sens),
            "specificity": float(spec),
        })

    df = pd.DataFrame(rows)

    # Deterministic tie-breaking:
    # 1) highest balanced accuracy;
    # 2) threshold closest to 0.5;
    # 3) smaller threshold.
    df["distance_to_0_5"] = np.abs(df["threshold"] - 0.5)
    best = (
        df.sort_values(
            [
                "balanced_accuracy",
                "distance_to_0_5",
                "threshold",
            ],
            ascending=[False, True, True],
            kind="stable",
        )
        .iloc[0]
    )

    return float(best["threshold"]), df.drop(
        columns=["distance_to_0_5"]
    )


def point_metrics(y, scores, threshold):
    pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    return {
        "auroc": safe_auroc(y, scores),
        "auprc": safe_auprc(y, scores),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, pred)
        ),
        "sensitivity": float(
            recall_score(
                y,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "specificity": float(
            recall_score(
                y,
                pred,
                pos_label=0,
                zero_division=0,
            )
        ),
        "f1_insufficient": float(
            f1_score(
                y,
                pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def stratified_bootstrap_cis(
    y,
    scores,
    threshold,
    n_boot,
    seed,
):
    """
    Resample sufficient and insufficient test examples separately so every
    bootstrap replicate preserves both classes.
    """
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)

    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)

    if not len(idx0) or not len(idx1):
        raise ValueError("Bootstrap requires both classes")

    rng = np.random.default_rng(seed)

    names = [
        "auroc",
        "auprc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_insufficient",
    ]
    values = {name: [] for name in names}

    for _ in range(n_boot):
        b0 = rng.choice(idx0, size=len(idx0), replace=True)
        b1 = rng.choice(idx1, size=len(idx1), replace=True)
        idx = np.concatenate([b0, b1])

        bm = point_metrics(
            y[idx],
            scores[idx],
            threshold,
        )

        for name in names:
            values[name].append(bm[name])

    out = {}
    for name in names:
        lo, hi = np.percentile(
            np.asarray(values[name], dtype=float),
            [2.5, 97.5],
        )
        out[f"{name}_ci95_low"] = float(lo)
        out[f"{name}_ci95_high"] = float(hi)

    return out


def train_one_analysis(
    model: str,
    analysis: str,
    cache,
    args,
    outdir: Path,
):
    outdir.mkdir(parents=True, exist_ok=True)

    X_train = cache["train"]["X"]
    y_train = cache["train"]["y"]
    X_val = cache["validation"]["X"]
    y_val = cache["validation"]["y"]
    X_test = cache["test"]["X"]
    y_test = cache["test"]["y"]

    if X_train.ndim != 3:
        raise RuntimeError(
            f"Expected X_train [n,layers,hidden], got {X_train.shape}"
        )

    n_layers = X_train.shape[1]
    if (
        X_val.shape[1] != n_layers
        or X_test.shape[1] != n_layers
    ):
        raise RuntimeError("Layer count differs across splits")

    layer_rows = []

    print(
        f"\nTraining {model}/{analysis}: "
        f"{n_layers} layer-wise probes"
    )

    for layer in range(n_layers):
        probe = make_probe(
            args.C,
            args.seed,
            args.max_iter,
        )
        probe.fit(
            X_train[:, layer, :],
            y_train,
        )

        p_val = probe.predict_proba(
            X_val[:, layer, :]
        )[:, 1]

        auroc = safe_auroc(
            y_val,
            p_val,
        )
        auprc = safe_auprc(
            y_val,
            p_val,
        )

        layer_rows.append({
            "layer_0based": int(layer),
            "layer_1based": int(layer + 1),
            "validation_auroc": auroc,
            "validation_auprc": auprc,
        })

        print(
            f"  layer {layer:2d} "
            f"(paper {layer+1:2d}) "
            f"val_AUROC={auroc:.6f} "
            f"val_AUPRC={auprc:.6f}"
        )

    layer_df = pd.DataFrame(layer_rows)
    layer_df.to_csv(
        outdir / "layer_validation.csv",
        index=False,
    )

    best_idx = pd.to_numeric(
        layer_df["validation_auroc"],
        errors="coerce",
    ).idxmax()

    selected_layer = int(
        layer_df.loc[best_idx, "layer_0based"]
    )

    # Refit selected layer on TRAIN only.
    probe = make_probe(
        args.C,
        args.seed,
        args.max_iter,
    )
    probe.fit(
        X_train[:, selected_layer, :],
        y_train,
    )

    p_val = probe.predict_proba(
        X_val[:, selected_layer, :]
    )[:, 1]

    threshold, threshold_df = choose_threshold(
        y_val,
        p_val,
    )
    threshold_df.to_csv(
        outdir / "threshold_validation.csv",
        index=False,
    )

    p_test = probe.predict_proba(
        X_test[:, selected_layer, :]
    )[:, 1]

    metrics = point_metrics(
        y_test,
        p_test,
        threshold,
    )

    ci = stratified_bootstrap_cis(
        y_test,
        p_test,
        threshold,
        args.bootstrap,
        args.bootstrap_seed,
    )
    metrics.update(ci)

    joblib.dump(
        probe,
        outdir / "selected_probe.joblib",
    )

    predictions = cache["test"]["manifest"].copy()
    predictions["analysis"] = analysis
    predictions["analysis_label"] = y_test
    predictions["probe_score"] = p_test
    predictions["prediction"] = (
        p_test >= threshold
    ).astype(int)
    predictions["selected_layer_0based"] = selected_layer
    predictions["selected_layer_1based"] = selected_layer + 1
    predictions["threshold"] = threshold
    predictions.to_csv(
        outdir / "test_predictions.csv",
        index=False,
    )

    meta = {
        "model": model,
        "analysis": analysis,
        "C": float(args.C),
        "class_weight": "balanced",
        "solver": "liblinear",
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "selected_layer_0based": int(selected_layer),
        "selected_layer_1based": int(selected_layer + 1),
        "selected_threshold": float(threshold),
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_class0": int((y_train == 0).sum()),
        "train_class1": int((y_train == 1).sum()),
        "validation_class0": int((y_val == 0).sum()),
        "validation_class1": int((y_val == 1).sum()),
        "test_class0": int((y_test == 0).sum()),
        "test_class1": int((y_test == 1).sum()),
        "layer_selection": "validation AUROC",
        "threshold_selection": "validation balanced accuracy",
        "test_used_for_selection": False,
        "bootstrap": int(args.bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        **metrics,
    }

    (
        outdir / "metrics.json"
    ).write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    summary = {
        "model": model,
        "analysis": analysis,
        **{
            k: v
            for k, v in meta.items()
            if k not in {"model", "analysis"}
        },
    }

    print()
    print(
        f"SELECTED {model}/{analysis}: "
        f"layer={selected_layer} (paper {selected_layer+1}), "
        f"threshold={threshold:.8f}"
    )
    print(
        f"TEST: AUROC={metrics['auroc']:.6f} "
        f"AUPRC={metrics['auprc']:.6f} "
        f"BA={metrics['balanced_accuracy']:.6f} "
        f"sens={metrics['sensitivity']:.6f} "
        f"spec={metrics['specificity']:.6f}"
    )
    print(
        f"AUROC 95% CI: "
        f"[{metrics['auroc_ci95_low']:.6f}, "
        f"{metrics['auroc_ci95_high']:.6f}]"
    )

    return summary


def compare_combined_to_frozen(
    project_root: Path,
    model: str,
    summary: dict,
    strict: bool,
):
    probe_dir = (
        project_root
        / "outputs"
        / "probes"
        / model
    )

    meta_path = probe_dir / "selected_probe_metadata.json"

    expected_layer = EXPECTED_MAIN_LAYERS_0BASED[model]
    observed_layer = int(
        summary["selected_layer_0based"]
    )

    print()
    print("COMBINED SANITY CHECK")
    print(
        f"  expected frozen layer: {expected_layer} "
        f"(paper {expected_layer+1})"
    )
    print(
        f"  rerun selected layer:  {observed_layer} "
        f"(paper {observed_layer+1})"
    )

    if meta_path.exists():
        meta = json.loads(
            meta_path.read_text(encoding="utf-8")
        )
        for key in [
            "test_auroc",
            "healthbench_test_auroc",
            "auroc",
        ]:
            if key in meta:
                print(
                    f"  frozen metadata {key}: "
                    f"{float(meta[key]):.6f}"
                )
                break

    print(
        f"  rerun test AUROC:      "
        f"{float(summary['auroc']):.6f}"
    )

    if strict and observed_layer != expected_layer:
        raise RuntimeError(
            f"{model}: combined sensitivity pipeline selected layer "
            f"{observed_layer}, expected frozen main layer {expected_layer}. "
            "Investigate activation loading/protocol before interpreting "
            "family-specific results."
        )

    if observed_layer != expected_layer:
        print(
            "  WARNING: selected layer differs from final frozen main probe. "
            "Do not interpret sensitivity results until this is understood."
        )
    else:
        print("  Layer reproduction: PASS")


def main():
    args = parse_args()

    root = args.project_root.resolve()
    manifest_path = (
        root
        / "data"
        / "manifests"
        / "healthbench_evidence_sufficiency_v1.csv"
    )
    output_root = (
        root
        / "outputs"
        / "sensitivity"
        / "healthbench_label_families"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = load_manifest(
        manifest_path
    )

    print_subset_audit(
        manifest
    )

    # Save exact subset manifests for reproducibility.
    subset_dir = output_root / "subset_manifests"
    subset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for analysis in [
        "combined",
        "context_seeking",
        "health_data_task",
    ]:
        x = build_analysis_manifest(
            manifest,
            analysis,
        )
        save = x.drop(
            columns=["__categories"],
            errors="ignore",
        )
        save.to_csv(
            subset_dir / f"{analysis}.csv",
            index=False,
        )

    if args.audit_only:
        print(
            "\nAudit-only requested. No probes were trained."
        )
        return

    all_summaries = []

    for model in args.models:
        print()
        print("#" * 88)
        print(f"MODEL: {model}")
        print("#" * 88)

        full_cache = load_model_cache(
            root,
            model,
            manifest,
        )

        model_summaries = {}

        for analysis in [
            "combined",
            "context_seeking",
            "health_data_task",
        ]:
            print()
            print("=" * 88)
            print(
                f"ANALYSIS: {analysis}"
            )
            print("=" * 88)

            analysis_manifest = build_analysis_manifest(
                manifest,
                analysis,
            )

            cache = subset_cache_for_analysis(
                full_cache,
                analysis_manifest,
            )

            outdir = (
                output_root
                / model
                / analysis
            )

            summary = train_one_analysis(
                model,
                analysis,
                cache,
                args,
                outdir,
            )

            all_summaries.append(
                summary
            )
            model_summaries[analysis] = summary

        compare_combined_to_frozen(
            root,
            model,
            model_summaries["combined"],
            args.strict_main_layer_check,
        )

        # Free the large activation arrays before loading next model.
        del full_cache

    summary_df = pd.DataFrame(
        all_summaries
    )

    summary_df.to_csv(
        output_root / "summary_all_models.csv",
        index=False,
    )

    # Compact paper/rebuttal table.
    paper_cols = [
        "model",
        "analysis",
        "n_train",
        "n_validation",
        "n_test",
        "selected_layer_1based",
        "auroc",
        "auroc_ci95_low",
        "auroc_ci95_high",
        "auprc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_insufficient",
    ]
    summary_df[paper_cols].to_csv(
        output_root / "paper_table.csv",
        index=False,
    )

    print()
    print("=" * 88)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 88)
    print(
        summary_df[paper_cols].to_string(
            index=False
        )
    )
    print()
    print(
        "Saved summary:",
        output_root / "summary_all_models.csv",
    )
    print(
        "Saved compact table:",
        output_root / "paper_table.csv",
    )
    print("=" * 88)


if __name__ == "__main__":
    main()
