# -*- coding: utf-8 -*-

"""
BRIDGE Revision - Linear Probe Training and Evaluation
======================================================

For each target model:

1. Load HealthBench TRAIN activations.
2. Train one L2 logistic regression probe at every layer.
3. Evaluate every layer on HealthBench VALIDATION.
4. Select the layer with highest validation AUROC.
5. Refit the selected-layer probe using HealthBench TRAIN only.
6. Select the classification threshold using validation
   balanced accuracy.
7. Freeze:
       - probe weights
       - selected layer
       - threshold
8. Only after freezing, evaluate on:
       - HealthBench Hard test
       - ClinDet external test
9. Compute:
       - AUROC
       - AUPRC
       - balanced accuracy
       - sensitivity for insufficient examples
       - specificity for sufficient examples
       - F1 for insufficient examples
       - 95 percent stratified-bootstrap confidence intervals
10. Report ClinDet subgroups:
       - Complete
       - Incomplete_Determinable
       - Incomplete_Undeterminable

Important label convention:

    y = 0 : evidence sufficient
    y = 1 : evidence insufficient

Important score terminology:

The logistic regression is trained with class_weight="balanced".
Therefore predict_proba() is used only as a monotonic
probe-derived insufficiency score.

It should NOT be described as a calibrated probability unless
calibration is separately demonstrated.

Expected input layout:

cache/activations/
    biomistral/
    openbiollm/
    ultramedical/
    medgemma/
    lingshu/

Output layout:

outputs/probes/
    summary.csv
    run_config.json

    biomistral/
        layer_validation.csv
        threshold_validation.csv
        selected_probe.joblib
        weights.npy
        cav.npy
        intercept.npy
        selected_probe_metadata.json
        healthbench_test_predictions.csv
        clindet_external_predictions.csv
        clindet_subgroups.csv
        metrics.json

    ...

Logs:

logs/probe_training_YYYYMMDD_HHMMSS.log

Usage:

python scripts/train_probes.py \
    --models all \
    --bootstrap 2000
"""

import argparse
import gc
import json
import logging
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_ACTIVATION_ROOT = Path(
    "cache/activations"
)

DEFAULT_OUTPUT_ROOT = Path(
    "outputs/probes"
)

DEFAULT_LOG_DIR = Path(
    "logs"
)


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

MODEL_SPECS = {
    "biomistral": {
        "num_layers": 32,
        "hidden_size": 4096,
    },

    "openbiollm": {
        "num_layers": 32,
        "hidden_size": 4096,
    },

    "ultramedical": {
        "num_layers": 32,
        "hidden_size": 4096,
    },

    "medgemma": {
        "num_layers": 34,
        "hidden_size": 2560,
    },

    "lingshu": {
        "num_layers": 28,
        "hidden_size": 3584,
    },
}


# ============================================================
# EXPECTED SPLIT COUNTS
# ============================================================

EXPECTED_COUNTS = {
    (
        "healthbench",
        "train",
    ): {
        0: 310,
        1: 182,
    },

    (
        "healthbench",
        "validation",
    ): {
        0: 78,
        1: 45,
    },

    (
        "healthbench",
        "test",
    ): {
        0: 54,
        1: 134,
    },

    (
        "clindet",
        "external_test",
    ): {
        0: 62,
        1: 32,
    },
}


# ============================================================
# PROBE CONFIGURATION
# ============================================================

DEFAULT_C = 1.0
DEFAULT_SEED = 42
DEFAULT_MAX_ITER = 10000
DEFAULT_BOOTSTRAP = 2000


# ============================================================
# LOGGING
# ============================================================

def setup_logging(log_dir):

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = (
        log_dir
        / f"probe_training_{timestamp}.log"
    )

    logger = logging.getLogger(
        "bridge_probe_training"
    )

    logger.setLevel(
        logging.INFO
    )

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(message)s"
    )

    console_handler = (
        logging.StreamHandler(
            sys.stdout
        )
    )

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    file_handler = (
        logging.FileHandler(
            log_path,
            mode="w",
            encoding="utf-8",
        )
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )

    return (
        logger,
        log_path,
    )


# ============================================================
# GENERAL HELPERS
# ============================================================

def to_json_safe(value):

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        return float(
            value
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    raise TypeError(
        f"Cannot JSON serialize {type(value)}"
    )


def class_counts(y):

    unique, counts = np.unique(
        y,
        return_counts=True,
    )

    result = {
        0: 0,
        1: 0,
    }

    for label, count in zip(
        unique,
        counts,
    ):

        result[
            int(label)
        ] = int(
            count
        )

    return result


# ============================================================
# LOAD ACTIVATIONS
# ============================================================

def load_activation_split(
    activation_root,
    model_name,
    dataset,
    split,
    expected_layers,
    expected_hidden,
    logger,
):
    """
    Load one model/dataset/split into memory.

    Returns:

        X:
            [N, layers, hidden]

        y:
            [N]

        metadata:
            list of dictionaries
    """

    directory = (
        activation_root
        / model_name
        / dataset
        / split
    )

    if not directory.exists():

        raise FileNotFoundError(
            f"Activation directory not found: "
            f"{directory}"
        )

    files = sorted(
        directory.glob(
            "*.pt"
        )
    )

    if not files:

        raise RuntimeError(
            f"No .pt files found in {directory}"
        )

    expected_label_counts = (
        EXPECTED_COUNTS[
            (
                dataset,
                split,
            )
        ]
    )

    expected_total = sum(
        expected_label_counts.values()
    )

    if len(files) != expected_total:

        raise RuntimeError(
            f"{model_name}/{dataset}/{split}: "
            f"expected {expected_total} files, "
            f"found {len(files)}"
        )

    activations = []
    labels = []
    metadata = []

    for path in files:

        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        required = {
            "activations",
            "example_id",
            "dataset",
            "split",
            "label",
            "model_name",
            "token_count",
        }

        missing = (
            required
            - set(
                obj.keys()
            )
        )

        if missing:

            raise RuntimeError(
                f"{path}: missing keys "
                f"{sorted(missing)}"
            )

        if (
            obj[
                "model_name"
            ]
            != model_name
        ):

            raise RuntimeError(
                f"Model metadata mismatch: {path}"
            )

        if (
            obj[
                "dataset"
            ]
            != dataset
        ):

            raise RuntimeError(
                f"Dataset metadata mismatch: {path}"
            )

        if (
            obj[
                "split"
            ]
            != split
        ):

            raise RuntimeError(
                f"Split metadata mismatch: {path}"
            )

        x = obj[
            "activations"
        ]

        expected_shape = (
            expected_layers,
            expected_hidden,
        )

        if tuple(
            x.shape
        ) != expected_shape:

            raise RuntimeError(
                f"{path}: expected activation "
                f"shape {expected_shape}, "
                f"got {tuple(x.shape)}"
            )

        if not torch.isfinite(
            x
        ).all():

            raise RuntimeError(
                f"Non-finite activation in {path}"
            )

        label = int(
            obj[
                "label"
            ]
        )

        if label not in {
            0,
            1,
        }:

            raise RuntimeError(
                f"Invalid label {label}: {path}"
            )

        activations.append(
            x.numpy().astype(
                np.float32,
                copy=False,
            )
        )

        labels.append(
            label
        )

        source_metadata = obj.get(
            "source_metadata",
            {},
        )

        if (
            source_metadata
            is None
        ):

            source_metadata = {}

        metadata.append(
            {
                "example_id":
                    str(
                        obj[
                            "example_id"
                        ]
                    ),

                "dataset":
                    dataset,

                "split":
                    split,

                "label":
                    label,

                "token_count":
                    int(
                        obj[
                            "token_count"
                        ]
                    ),

                "source_metadata":
                    source_metadata,
            }
        )

    X = np.stack(
        activations,
        axis=0,
    )

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    observed_counts = (
        class_counts(
            y
        )
    )

    if (
        observed_counts
        != expected_label_counts
    ):

        raise RuntimeError(
            f"{model_name}/{dataset}/{split}: "
            f"expected class counts "
            f"{expected_label_counts}, "
            f"found {observed_counts}"
        )

    logger.info(
        f"Loaded "
        f"{model_name}/"
        f"{dataset}/"
        f"{split}: "
        f"X={X.shape}, "
        f"class0={observed_counts[0]}, "
        f"class1={observed_counts[1]}"
    )

    return (
        X,
        y,
        metadata,
    )


# ============================================================
# PROBE TRAINING
# ============================================================

def make_probe(
    C,
    seed,
    max_iter,
):
    """
    The primary probe follows the submitted method:

        L2 logistic regression
        C = 1.0 by default
        class_weight = balanced

    No feature standardization is performed.

    This keeps the learned coefficient vector directly in the
    hidden-state coordinate system, which is useful later for
    CAV/mechanistic analysis.
    """

    return LogisticRegression(
        penalty="l2",
        C=C,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
        max_iter=max_iter,
        tol=1e-5,
    )


def fit_probe(
    X,
    y,
    C,
    seed,
    max_iter,
    logger=None,
):
    """
    Fit probe and automatically retry with a larger iteration
    budget if sklearn reports non-convergence.
    """

    probe = make_probe(
        C=C,
        seed=seed,
        max_iter=max_iter,
    )

    with warnings.catch_warnings(
        record=True
    ) as caught:

        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        probe.fit(
            X,
            y,
        )

    convergence_warnings = [
        warning
        for warning
        in caught
        if issubclass(
            warning.category,
            ConvergenceWarning,
        )
    ]

    if convergence_warnings:

        if logger is not None:

            logger.info(
                "  Convergence warning detected. "
                "Retrying with larger max_iter."
            )

        retry_max_iter = (
            max_iter
            * 3
        )

        probe = make_probe(
            C=C,
            seed=seed,
            max_iter=retry_max_iter,
        )

        probe.fit(
            X,
            y,
        )

    return probe


def get_probe_scores(
    probe,
    X,
):
    """
    Return sigmoid-transformed probe scores for class y=1.

    Because class_weight='balanced' is used, these should be
    described as probe-derived insufficiency scores, not
    calibrated probabilities.
    """

    classes = list(
        probe.classes_
    )

    if 1 not in classes:

        raise RuntimeError(
            "Probe classes do not contain positive class 1."
        )

    positive_column = (
        classes.index(
            1
        )
    )

    return (
        probe.predict_proba(
            X
        )[
            :,
            positive_column
        ]
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y,
    scores,
    threshold,
):
    """
    Compute final binary detection metrics.
    """

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    predictions = (
        scores
        >= threshold
    ).astype(
        np.int64
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            predictions,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )

    sensitivity = (
        tp
        / (
            tp
            + fn
        )
        if (
            tp + fn
        ) > 0
        else np.nan
    )

    specificity = (
        tn
        / (
            tn
            + fp
        )
        if (
            tn + fp
        ) > 0
        else np.nan
    )

    prevalence = float(
        np.mean(
            y == 1
        )
    )

    result = {
        "n":
            int(
                len(y)
            ),

        "positive_prevalence":
            prevalence,

        "auprc_no_skill":
            prevalence,

        "auroc":
            float(
                roc_auc_score(
                    y,
                    scores,
                )
            ),

        "auprc":
            float(
                average_precision_score(
                    y,
                    scores,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y,
                    predictions,
                )
            ),

        "sensitivity":
            float(
                sensitivity
            ),

        "specificity":
            float(
                specificity
            ),

        "f1":
            float(
                f1_score(
                    y,
                    predictions,
                    pos_label=1,
                    zero_division=0,
                )
            ),

        "tn":
            int(
                tn
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "tp":
            int(
                tp
            ),

        "threshold":
            float(
                threshold
            ),
    }

    return result


# ============================================================
# LAYER SELECTION
# ============================================================

def train_layerwise_probes(
    X_train,
    y_train,
    X_val,
    y_val,
    C,
    seed,
    max_iter,
    logger,
):
    """
    Train one probe at every layer.

    IMPORTANT:
    Layer selection uses VALIDATION AUROC only.
    """

    num_layers = (
        X_train.shape[1]
    )

    rows = []

    logger.info("")
    logger.info(
        "Training one probe per layer..."
    )

    for layer_idx in range(
        num_layers
    ):

        start = time.time()

        Xtr = (
            X_train[
                :,
                layer_idx,
                :
            ]
            .astype(
                np.float64,
                copy=False,
            )
        )

        Xv = (
            X_val[
                :,
                layer_idx,
                :
            ]
            .astype(
                np.float64,
                copy=False,
            )
        )

        probe = fit_probe(
            X=Xtr,
            y=y_train,
            C=C,
            seed=seed,
            max_iter=max_iter,
            logger=logger,
        )

        val_scores = (
            get_probe_scores(
                probe,
                Xv,
            )
        )

        val_auroc = float(
            roc_auc_score(
                y_val,
                val_scores,
            )
        )

        val_auprc = float(
            average_precision_score(
                y_val,
                val_scores,
            )
        )

        elapsed = (
            time.time()
            - start
        )

        rows.append(
            {
                "layer_index_0based":
                    layer_idx,

                "layer_number_1based":
                    layer_idx + 1,

                "validation_auroc":
                    val_auroc,

                "validation_auprc":
                    val_auprc,

                "training_seconds":
                    elapsed,
            }
        )

        logger.info(
            f"  layer "
            f"{layer_idx:2d} "
            f"(paper layer {layer_idx + 1:2d}) "
            f"AUROC={val_auroc:.4f} "
            f"AUPRC={val_auprc:.4f} "
            f"time={elapsed:.2f}s"
        )

        del probe
        gc.collect()

    layer_df = pd.DataFrame(
        rows
    )

    # Selection criterion is AUROC only.
    #
    # If two layers have exactly equal AUROC, choose the lower
    # layer index deterministically.
    best_row = (
        layer_df
        .sort_values(
            by=[
                "validation_auroc",
                "layer_index_0based",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .iloc[0]
    )

    selected_layer = int(
        best_row[
            "layer_index_0based"
        ]
    )

    logger.info("")
    logger.info(
        "Selected layer by validation AUROC:"
    )

    logger.info(
        f"  0-based layer index: "
        f"{selected_layer}"
    )

    logger.info(
        f"  1-based paper layer: "
        f"{selected_layer + 1}"
    )

    logger.info(
        f"  validation AUROC: "
        f"{best_row['validation_auroc']:.6f}"
    )

    return (
        selected_layer,
        layer_df,
    )


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def make_threshold_candidates(
    scores,
):
    """
    Construct all meaningful threshold regions from validation
    scores.

    Predictions are:

        y_hat = 1 if score >= threshold

    We use:
        - threshold 0.0
        - midpoint between every pair of unique scores
        - threshold 1.0

    This yields all practically distinct classification
    partitions for scores in [0, 1].
    """

    unique_scores = np.unique(
        np.asarray(
            scores,
            dtype=np.float64,
        )
    )

    unique_scores.sort()

    candidates = [
        0.0
    ]

    if len(
        unique_scores
    ) > 1:

        midpoints = (
            unique_scores[:-1]
            + unique_scores[1:]
        ) / 2.0

        candidates.extend(
            midpoints.tolist()
        )

    candidates.append(
        1.0
    )

    candidates = np.asarray(
        sorted(
            set(
                float(x)
                for x
                in candidates
            )
        ),
        dtype=np.float64,
    )

    return candidates


def select_threshold(
    y_val,
    val_scores,
    logger,
):
    """
    Select threshold by validation balanced accuracy.

    Tie breaking:
        1. higher balanced accuracy
        2. threshold closer to 0.5
        3. smaller threshold

    No test data are used.
    """

    candidates = (
        make_threshold_candidates(
            val_scores
        )
    )

    rows = []

    for threshold in candidates:

        predictions = (
            val_scores
            >= threshold
        ).astype(
            np.int64
        )

        tn, fp, fn, tp = (
            confusion_matrix(
                y_val,
                predictions,
                labels=[
                    0,
                    1,
                ],
            )
            .ravel()
        )

        sensitivity = (
            tp
            / (
                tp + fn
            )
        )

        specificity = (
            tn
            / (
                tn + fp
            )
        )

        balanced_acc = (
            (
                sensitivity
                + specificity
            )
            / 2.0
        )

        f1 = f1_score(
            y_val,
            predictions,
            pos_label=1,
            zero_division=0,
        )

        rows.append(
            {
                "threshold":
                    float(
                        threshold
                    ),

                "balanced_accuracy":
                    float(
                        balanced_acc
                    ),

                "sensitivity":
                    float(
                        sensitivity
                    ),

                "specificity":
                    float(
                        specificity
                    ),

                "f1":
                    float(
                        f1
                    ),

                "tn":
                    int(
                        tn
                    ),

                "fp":
                    int(
                        fp
                    ),

                "fn":
                    int(
                        fn
                    ),

                "tp":
                    int(
                        tp
                    ),
            }
        )

    threshold_df = pd.DataFrame(
        rows
    )

    threshold_df[
        "distance_from_0.5"
    ] = np.abs(
        threshold_df[
            "threshold"
        ]
        - 0.5
    )

    best = (
        threshold_df
        .sort_values(
            by=[
                "balanced_accuracy",
                "distance_from_0.5",
                "threshold",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .iloc[0]
    )

    selected_threshold = float(
        best[
            "threshold"
        ]
    )

    threshold_df[
        "selected"
    ] = (
        threshold_df[
            "threshold"
        ]
        == selected_threshold
    )

    logger.info("")
    logger.info(
        "Selected threshold by validation "
        "balanced accuracy:"
    )

    logger.info(
        f"  threshold: "
        f"{selected_threshold:.8f}"
    )

    logger.info(
        f"  balanced accuracy: "
        f"{best['balanced_accuracy']:.6f}"
    )

    logger.info(
        f"  sensitivity: "
        f"{best['sensitivity']:.6f}"
    )

    logger.info(
        f"  specificity: "
        f"{best['specificity']:.6f}"
    )

    return (
        selected_threshold,
        threshold_df,
    )


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

BOOTSTRAP_METRICS = [
    "auroc",
    "auprc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1",
]


def stratified_bootstrap_ci(
    y,
    scores,
    threshold,
    n_bootstrap,
    seed,
):
    """
    Stratified bootstrap.

    Positive and negative class counts are preserved in every
    bootstrap sample.

    This is especially useful because AUPRC depends on class
    prevalence.
    """

    y = np.asarray(
        y,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    positive_indices = np.where(
        y == 1
    )[0]

    negative_indices = np.where(
        y == 0
    )[0]

    if (
        len(
            positive_indices
        ) == 0
        or
        len(
            negative_indices
        ) == 0
    ):

        raise RuntimeError(
            "Bootstrap requires both classes."
        )

    rng = (
        np.random.default_rng(
            seed
        )
    )

    samples = {
        metric: []
        for metric
        in BOOTSTRAP_METRICS
    }

    for _ in range(
        n_bootstrap
    ):

        sampled_positive = (
            rng.choice(
                positive_indices,
                size=len(
                    positive_indices
                ),
                replace=True,
            )
        )

        sampled_negative = (
            rng.choice(
                negative_indices,
                size=len(
                    negative_indices
                ),
                replace=True,
            )
        )

        bootstrap_indices = (
            np.concatenate(
                [
                    sampled_negative,
                    sampled_positive,
                ]
            )
        )

        bootstrap_y = y[
            bootstrap_indices
        ]

        bootstrap_scores = scores[
            bootstrap_indices
        ]

        metrics = compute_metrics(
            y=bootstrap_y,
            scores=bootstrap_scores,
            threshold=threshold,
        )

        for metric in (
            BOOTSTRAP_METRICS
        ):

            samples[
                metric
            ].append(
                metrics[
                    metric
                ]
            )

    intervals = {}

    for metric in (
        BOOTSTRAP_METRICS
    ):

        values = np.asarray(
            samples[
                metric
            ],
            dtype=np.float64,
        )

        intervals[
            metric
        ] = {
            "lower":
                float(
                    np.percentile(
                        values,
                        2.5,
                    )
                ),

            "upper":
                float(
                    np.percentile(
                        values,
                        97.5,
                    )
                ),
        }

    return intervals


# ============================================================
# FINAL DATASET EVALUATION
# ============================================================

def evaluate_dataset(
    probe,
    X,
    y,
    selected_layer,
    threshold,
    n_bootstrap,
    bootstrap_seed,
):
    """
    Evaluate one frozen probe on one frozen test dataset.
    """

    X_selected = (
        X[
            :,
            selected_layer,
            :
        ]
        .astype(
            np.float64,
            copy=False,
        )
    )

    scores = (
        get_probe_scores(
            probe,
            X_selected,
        )
    )

    metrics = (
        compute_metrics(
            y=y,
            scores=scores,
            threshold=threshold,
        )
    )

    ci = (
        stratified_bootstrap_ci(
            y=y,
            scores=scores,
            threshold=threshold,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
    )

    metrics[
        "confidence_intervals_95"
    ] = ci

    predictions = (
        scores
        >= threshold
    ).astype(
        np.int64
    )

    return (
        metrics,
        scores,
        predictions,
    )


# ============================================================
# PREDICTION CSV
# ============================================================

def build_prediction_dataframe(
    metadata,
    scores,
    predictions,
    selected_layer,
    threshold,
):
    rows = []

    for (
        item,
        score,
        prediction,
    ) in zip(
        metadata,
        scores,
        predictions,
    ):

        source_metadata = item.get(
            "source_metadata",
            {},
        )

        if (
            source_metadata
            is None
        ):

            source_metadata = {}

        rows.append(
            {
                "example_id":
                    item[
                        "example_id"
                    ],

                "dataset":
                    item[
                        "dataset"
                    ],

                "split":
                    item[
                        "split"
                    ],

                "label":
                    int(
                        item[
                            "label"
                        ]
                    ),

                "probe_score":
                    float(
                        score
                    ),

                "prediction":
                    int(
                        prediction
                    ),

                "correct":
                    int(
                        prediction
                        == item[
                            "label"
                        ]
                    ),

                "selected_layer_0based":
                    selected_layer,

                "selected_layer_1based":
                    selected_layer
                    + 1,

                "threshold":
                    float(
                        threshold
                    ),

                "token_count":
                    int(
                        item[
                            "token_count"
                        ]
                    ),

                "information_condition":
                    source_metadata.get(
                        "information_condition",
                        "",
                    ),

                "source_metadata_json":
                    json.dumps(
                        source_metadata,
                        ensure_ascii=True,
                        default=str,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# WILSON CONFIDENCE INTERVAL
# ============================================================

def wilson_interval(
    successes,
    n,
    z=1.96,
):
    """
    Wilson 95 percent interval for a binary proportion.
    """

    if n == 0:

        return (
            np.nan,
            np.nan,
        )

    p = (
        successes
        / n
    )

    denominator = (
        1.0
        + (
            z * z
            / n
        )
    )

    center = (
        p
        + (
            z * z
            / (
                2.0 * n
            )
        )
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            (
                p
                * (
                    1.0 - p
                )
                / n
            )
            + (
                z
                * z
                / (
                    4.0
                    * n
                    * n
                )
            )
        )
        / denominator
    )

    return (
        max(
            0.0,
            center
            - half_width,
        ),

        min(
            1.0,
            center
            + half_width,
        ),
    )


# ============================================================
# CLINDET SUBGROUP ANALYSIS
# ============================================================

def build_clindet_subgroups(
    prediction_df,
):
    """
    Each ClinDet information-condition subgroup contains only
    one binary sufficiency label.

    Therefore AUROC/AUPRC are not defined inside an individual
    subgroup.

    We report:

        N
        true binary label
        predicted-insufficient rate
        mean probe score
        median probe score
        correctness
        Wilson 95 percent CI for correctness

    Complete:
        label 0

    Incomplete_Determinable:
        label 0

    Incomplete_Undeterminable:
        label 1
    """

    expected_groups = [
        "Complete",
        "Incomplete_Determinable",
        "Incomplete_Undeterminable",
    ]

    rows = []

    for group in (
        expected_groups
    ):

        subset = prediction_df[
            prediction_df[
                "information_condition"
            ]
            == group
        ].copy()

        if len(
            subset
        ) == 0:

            rows.append(
                {
                    "information_condition":
                        group,

                    "n":
                        0,
                }
            )

            continue

        labels = subset[
            "label"
        ].to_numpy()

        unique_labels = np.unique(
            labels
        )

        if len(
            unique_labels
        ) != 1:

            raise RuntimeError(
                f"ClinDet subgroup {group} "
                "contains multiple binary labels."
            )

        true_label = int(
            unique_labels[0]
        )

        predictions = subset[
            "prediction"
        ].to_numpy()

        scores = subset[
            "probe_score"
        ].to_numpy()

        correct = int(
            np.sum(
                predictions
                == labels
            )
        )

        n = int(
            len(
                subset
            )
        )

        correct_rate = (
            correct
            / n
        )

        ci_low, ci_high = (
            wilson_interval(
                successes=correct,
                n=n,
            )
        )

        predicted_insufficient_rate = float(
            np.mean(
                predictions
                == 1
            )
        )

        if true_label == 0:

            class_metric_name = (
                "specificity"
            )

        else:

            class_metric_name = (
                "sensitivity"
            )

        rows.append(
            {
                "information_condition":
                    group,

                "n":
                    n,

                "true_binary_label":
                    true_label,

                "predicted_insufficient_rate":
                    predicted_insufficient_rate,

                "mean_probe_score":
                    float(
                        np.mean(
                            scores
                        )
                    ),

                "median_probe_score":
                    float(
                        np.median(
                            scores
                        )
                    ),

                "correct_rate":
                    float(
                        correct_rate
                    ),

                "correct_rate_ci95_low":
                    float(
                        ci_low
                    ),

                "correct_rate_ci95_high":
                    float(
                        ci_high
                    ),

                "class_metric":
                    class_metric_name,

                "class_metric_value":
                    float(
                        correct_rate
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SAVE FROZEN PROBE
# ============================================================

def save_probe_artifacts(
    model_output_dir,
    probe,
    model_name,
    selected_layer,
    selected_threshold,
    C,
    seed,
):
    """
    Save the exact fitted probe and CAV.

    Because there is no feature standardization:

        CAV = normalized logistic-regression weight vector
    """

    model_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    weights = (
        probe.coef_
        .reshape(-1)
        .astype(
            np.float32
        )
    )

    intercept = np.asarray(
        probe.intercept_,
        dtype=np.float32,
    )

    weight_norm = float(
        np.linalg.norm(
            weights
        )
    )

    if (
        not np.isfinite(
            weight_norm
        )
        or
        weight_norm <= 0.0
    ):

        raise RuntimeError(
            "Invalid probe weight norm."
        )

    cav = (
        weights
        / weight_norm
    ).astype(
        np.float32
    )

    np.save(
        model_output_dir
        / "weights.npy",
        weights,
    )

    np.save(
        model_output_dir
        / "intercept.npy",
        intercept,
    )

    np.save(
        model_output_dir
        / "cav.npy",
        cav,
    )

    joblib.dump(
        {
            "probe":
                probe,

            "model_name":
                model_name,

            "selected_layer_0based":
                selected_layer,

            "selected_layer_1based":
                selected_layer + 1,

            "threshold":
                selected_threshold,

            "C":
                C,

            "class_weight":
                "balanced",

            "penalty":
                "l2",

            "solver":
                "liblinear",

            "seed":
                seed,
        },

        model_output_dir
        / "selected_probe.joblib",
    )

    metadata = {
        "model_name":
            model_name,

        "selected_layer_0based":
            selected_layer,

        "selected_layer_1based":
            selected_layer + 1,

        "threshold":
            float(
                selected_threshold
            ),

        "C":
            float(
                C
            ),

        "class_weight":
            "balanced",

        "penalty":
            "l2",

        "solver":
            "liblinear",

        "seed":
            int(
                seed
            ),

        "weight_norm":
            weight_norm,

        "score_description":
            (
                "Sigmoid-transformed logistic regression "
                "score for evidence insufficiency. "
                "Not treated as a calibrated probability "
                "because balanced class weighting changes "
                "the effective training prior."
            ),

        "cav_description":
            (
                "L2-normalized logistic regression "
                "coefficient vector in the original "
                "hidden-state coordinate system."
            ),
    }

    with open(
        model_output_dir
        / "selected_probe_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            default=to_json_safe,
        )


# ============================================================
# FORMAT SUMMARY ROW
# ============================================================

def make_summary_row(
    model_name,
    dataset_name,
    selected_layer,
    threshold,
    metrics,
    validation_auroc,
    validation_balanced_accuracy,
):
    row = {
        "model":
            model_name,

        "dataset":
            dataset_name,

        "selected_layer_0based":
            selected_layer,

        "selected_layer_1based":
            selected_layer
            + 1,

        "threshold":
            threshold,

        "validation_selected_layer_auroc":
            validation_auroc,

        "validation_threshold_balanced_accuracy":
            validation_balanced_accuracy,

        "n":
            metrics[
                "n"
            ],

        "positive_prevalence":
            metrics[
                "positive_prevalence"
            ],

        "auprc_no_skill":
            metrics[
                "auprc_no_skill"
            ],

        "auroc":
            metrics[
                "auroc"
            ],

        "auprc":
            metrics[
                "auprc"
            ],

        "balanced_accuracy":
            metrics[
                "balanced_accuracy"
            ],

        "sensitivity":
            metrics[
                "sensitivity"
            ],

        "specificity":
            metrics[
                "specificity"
            ],

        "f1":
            metrics[
                "f1"
            ],

        "tn":
            metrics[
                "tn"
            ],

        "fp":
            metrics[
                "fp"
            ],

        "fn":
            metrics[
                "fn"
            ],

        "tp":
            metrics[
                "tp"
            ],
    }

    ci = metrics[
        "confidence_intervals_95"
    ]

    for metric_name in (
        BOOTSTRAP_METRICS
    ):

        row[
            f"{metric_name}_ci95_low"
        ] = (
            ci[
                metric_name
            ][
                "lower"
            ]
        )

        row[
            f"{metric_name}_ci95_high"
        ] = (
            ci[
                metric_name
            ][
                "upper"
            ]
        )

    return row


# ============================================================
# RUN ONE MODEL
# ============================================================

def run_model(
    model_name,
    activation_root,
    output_root,
    C,
    seed,
    max_iter,
    n_bootstrap,
    logger,
):
    """
    Full leakage-safe pipeline for one target model.
    """

    model_spec = (
        MODEL_SPECS[
            model_name
        ]
    )

    num_layers = int(
        model_spec[
            "num_layers"
        ]
    )

    hidden_size = int(
        model_spec[
            "hidden_size"
        ]
    )

    model_output_dir = (
        output_root
        / model_name
    )

    model_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("")
    logger.info(
        "=" * 78
    )

    logger.info(
        f"MODEL: {model_name}"
    )

    logger.info(
        "=" * 78
    )

    logger.info(
        f"Expected layers: {num_layers}"
    )

    logger.info(
        f"Hidden size:     {hidden_size}"
    )

    # ========================================================
    # PHASE 1:
    # Load TRAIN and VALIDATION only.
    #
    # Test data are intentionally not loaded yet.
    # ========================================================

    logger.info("")
    logger.info(
        "PHASE 1 - TRAIN/VALIDATION ONLY"
    )

    X_train, y_train, _ = (
        load_activation_split(
            activation_root=activation_root,
            model_name=model_name,
            dataset="healthbench",
            split="train",
            expected_layers=num_layers,
            expected_hidden=hidden_size,
            logger=logger,
        )
    )

    X_val, y_val, _ = (
        load_activation_split(
            activation_root=activation_root,
            model_name=model_name,
            dataset="healthbench",
            split="validation",
            expected_layers=num_layers,
            expected_hidden=hidden_size,
            logger=logger,
        )
    )

    # ========================================================
    # Layer selection
    # ========================================================

    (
        selected_layer,
        layer_df,
    ) = (
        train_layerwise_probes(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            C=C,
            seed=seed,
            max_iter=max_iter,
            logger=logger,
        )
    )

    layer_df.to_csv(
        model_output_dir
        / "layer_validation.csv",
        index=False,
    )

    selected_layer_row = (
        layer_df[
            layer_df[
                "layer_index_0based"
            ]
            == selected_layer
        ]
        .iloc[0]
    )

    validation_selected_auroc = float(
        selected_layer_row[
            "validation_auroc"
        ]
    )

    # ========================================================
    # Refit selected-layer probe on TRAIN only
    # ========================================================

    logger.info("")
    logger.info(
        "Refitting selected-layer probe on "
        "HealthBench TRAIN only..."
    )

    selected_probe = fit_probe(
        X=(
            X_train[
                :,
                selected_layer,
                :
            ]
            .astype(
                np.float64,
                copy=False,
            )
        ),
        y=y_train,
        C=C,
        seed=seed,
        max_iter=max_iter,
        logger=logger,
    )

    val_scores = (
        get_probe_scores(
            selected_probe,
            X_val[
                :,
                selected_layer,
                :
            ].astype(
                np.float64,
                copy=False,
            ),
        )
    )

    # ========================================================
    # Threshold selection
    # ========================================================

    (
        selected_threshold,
        threshold_df,
    ) = (
        select_threshold(
            y_val=y_val,
            val_scores=val_scores,
            logger=logger,
        )
    )

    threshold_df.to_csv(
        model_output_dir
        / "threshold_validation.csv",
        index=False,
    )

    selected_threshold_row = (
        threshold_df[
            threshold_df[
                "selected"
            ]
        ]
        .iloc[0]
    )

    validation_threshold_balanced_accuracy = float(
        selected_threshold_row[
            "balanced_accuracy"
        ]
    )

    # ========================================================
    # Freeze and save probe BEFORE loading test data
    # ========================================================

    save_probe_artifacts(
        model_output_dir=model_output_dir,
        probe=selected_probe,
        model_name=model_name,
        selected_layer=selected_layer,
        selected_threshold=selected_threshold,
        C=C,
        seed=seed,
    )

    logger.info("")
    logger.info(
        "Probe, layer, and threshold are now FROZEN."
    )

    logger.info(
        "No test labels were used for fitting, "
        "layer selection, or threshold selection."
    )

    # ========================================================
    # PHASE 2:
    # Load final test sets only after freeze.
    # ========================================================

    logger.info("")
    logger.info(
        "PHASE 2 - FROZEN TEST EVALUATION"
    )

    X_test, y_test, test_metadata = (
        load_activation_split(
            activation_root=activation_root,
            model_name=model_name,
            dataset="healthbench",
            split="test",
            expected_layers=num_layers,
            expected_hidden=hidden_size,
            logger=logger,
        )
    )

    X_clindet, y_clindet, clindet_metadata = (
        load_activation_split(
            activation_root=activation_root,
            model_name=model_name,
            dataset="clindet",
            split="external_test",
            expected_layers=num_layers,
            expected_hidden=hidden_size,
            logger=logger,
        )
    )

    # ========================================================
    # HealthBench Hard
    # ========================================================

    logger.info("")
    logger.info(
        "Evaluating HealthBench Hard..."
    )

    (
        healthbench_metrics,
        healthbench_scores,
        healthbench_predictions,
    ) = (
        evaluate_dataset(
            probe=selected_probe,
            X=X_test,
            y=y_test,
            selected_layer=selected_layer,
            threshold=selected_threshold,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=seed,
        )
    )

    healthbench_prediction_df = (
        build_prediction_dataframe(
            metadata=test_metadata,
            scores=healthbench_scores,
            predictions=healthbench_predictions,
            selected_layer=selected_layer,
            threshold=selected_threshold,
        )
    )

    healthbench_prediction_df.to_csv(
        model_output_dir
        / "healthbench_test_predictions.csv",
        index=False,
    )

    # ========================================================
    # ClinDet
    # ========================================================

    logger.info(
        "Evaluating ClinDet external test..."
    )

    (
        clindet_metrics,
        clindet_scores,
        clindet_predictions,
    ) = (
        evaluate_dataset(
            probe=selected_probe,
            X=X_clindet,
            y=y_clindet,
            selected_layer=selected_layer,
            threshold=selected_threshold,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=(
                seed
                + 10000
            ),
        )
    )

    clindet_prediction_df = (
        build_prediction_dataframe(
            metadata=clindet_metadata,
            scores=clindet_scores,
            predictions=clindet_predictions,
            selected_layer=selected_layer,
            threshold=selected_threshold,
        )
    )

    clindet_prediction_df.to_csv(
        model_output_dir
        / "clindet_external_predictions.csv",
        index=False,
    )

    # ========================================================
    # ClinDet subgroup analysis
    # ========================================================

    clindet_subgroup_df = (
        build_clindet_subgroups(
            clindet_prediction_df
        )
    )

    clindet_subgroup_df.to_csv(
        model_output_dir
        / "clindet_subgroups.csv",
        index=False,
    )

    # ========================================================
    # Save metrics
    # ========================================================

    model_metrics = {
        "model":
            model_name,

        "selected_layer_0based":
            selected_layer,

        "selected_layer_1based":
            selected_layer + 1,

        "validation_selected_layer_auroc":
            validation_selected_auroc,

        "selected_threshold":
            selected_threshold,

        "validation_threshold_balanced_accuracy":
            validation_threshold_balanced_accuracy,

        "healthbench_hard":
            healthbench_metrics,

        "clindet_external":
            clindet_metrics,

        "bootstrap_resamples":
            n_bootstrap,

        "bootstrap_type":
            (
                "stratified nonparametric bootstrap "
                "with fixed class counts"
            ),
    }

    with open(
        model_output_dir
        / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            model_metrics,
            f,
            indent=2,
            default=to_json_safe,
        )

    # ========================================================
    # Log concise results
    # ========================================================

    logger.info("")
    logger.info(
        "FINAL RESULTS"
    )

    logger.info(
        "-" * 78
    )

    logger.info(
        "HealthBench Hard:"
    )

    logger.info(
        f"  AUROC:             "
        f"{healthbench_metrics['auroc']:.4f}"
    )

    logger.info(
        f"  AUPRC:             "
        f"{healthbench_metrics['auprc']:.4f} "
        f"(no-skill={healthbench_metrics['auprc_no_skill']:.4f})"
    )

    logger.info(
        f"  Balanced accuracy: "
        f"{healthbench_metrics['balanced_accuracy']:.4f}"
    )

    logger.info(
        f"  Sensitivity:       "
        f"{healthbench_metrics['sensitivity']:.4f}"
    )

    logger.info(
        f"  Specificity:       "
        f"{healthbench_metrics['specificity']:.4f}"
    )

    logger.info(
        f"  F1 insufficient:   "
        f"{healthbench_metrics['f1']:.4f}"
    )

    logger.info("")
    logger.info(
        "ClinDet external:"
    )

    logger.info(
        f"  AUROC:             "
        f"{clindet_metrics['auroc']:.4f}"
    )

    logger.info(
        f"  AUPRC:             "
        f"{clindet_metrics['auprc']:.4f} "
        f"(no-skill={clindet_metrics['auprc_no_skill']:.4f})"
    )

    logger.info(
        f"  Balanced accuracy: "
        f"{clindet_metrics['balanced_accuracy']:.4f}"
    )

    logger.info(
        f"  Sensitivity:       "
        f"{clindet_metrics['sensitivity']:.4f}"
    )

    logger.info(
        f"  Specificity:       "
        f"{clindet_metrics['specificity']:.4f}"
    )

    logger.info(
        f"  F1 insufficient:   "
        f"{clindet_metrics['f1']:.4f}"
    )

    logger.info("")
    logger.info(
        "ClinDet subgroups:"
    )

    for _, row in (
        clindet_subgroup_df
        .iterrows()
    ):

        if int(
            row[
                "n"
            ]
        ) == 0:

            logger.info(
                f"  {row['information_condition']}: "
                "NO EXAMPLES"
            )

            continue

        logger.info(
            f"  {row['information_condition']}: "
            f"n={int(row['n'])}, "
            f"correct={row['correct_rate']:.4f}, "
            f"mean_score={row['mean_probe_score']:.4f}"
        )

    # ========================================================
    # Summary rows
    # ========================================================

    summary_rows = [
        make_summary_row(
            model_name=model_name,
            dataset_name="healthbench_hard",
            selected_layer=selected_layer,
            threshold=selected_threshold,
            metrics=healthbench_metrics,
            validation_auroc=validation_selected_auroc,
            validation_balanced_accuracy=
                validation_threshold_balanced_accuracy,
        ),

        make_summary_row(
            model_name=model_name,
            dataset_name="clindet_external",
            selected_layer=selected_layer,
            threshold=selected_threshold,
            metrics=clindet_metrics,
            validation_auroc=validation_selected_auroc,
            validation_balanced_accuracy=
                validation_threshold_balanced_accuracy,
        ),
    ]

    # Cleanup
    del X_train
    del X_val
    del X_test
    del X_clindet
    del selected_probe

    gc.collect()

    return summary_rows


# ============================================================
# MODEL SELECTION CLI
# ============================================================

def parse_model_selection(
    requested,
):
    requested = list(
        requested
    )

    if requested == [
        "all"
    ]:

        return list(
            MODEL_SPECS.keys()
        )

    unknown = (
        set(
            requested
        )
        - set(
            MODEL_SPECS.keys()
        )
    )

    if unknown:

        raise ValueError(
            "Unknown model(s): "
            + ", ".join(
                sorted(
                    unknown
                )
            )
        )

    return requested


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate BRIDGE "
            "evidence-insufficiency probes."
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "all"
        ],
    )

    parser.add_argument(
        "--activation-root",
        default=str(
            DEFAULT_ACTIVATION_ROOT
        ),
    )

    parser.add_argument(
        "--output-root",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    parser.add_argument(
        "--log-dir",
        default=str(
            DEFAULT_LOG_DIR
        ),
    )

    parser.add_argument(
        "--C",
        type=float,
        default=DEFAULT_C,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
    )

    args = (
        parser.parse_args()
    )

    activation_root = Path(
        args.activation_root
    )

    output_root = Path(
        args.output_root
    )

    log_dir = Path(
        args.log_dir
    )

    selected_models = (
        parse_model_selection(
            args.models
        )
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger, log_path = (
        setup_logging(
            log_dir
        )
    )

    # ========================================================
    # Global run information
    # ========================================================

    logger.info(
        "=" * 78
    )

    logger.info(
        "BRIDGE LINEAR PROBE TRAINING"
    )

    logger.info(
        "=" * 78
    )

    logger.info(
        f"Start time:       {datetime.now()}"
    )

    logger.info(
        f"Activation root:  "
        f"{activation_root.resolve()}"
    )

    logger.info(
        f"Output root:      "
        f"{output_root.resolve()}"
    )

    logger.info(
        f"Log file:         "
        f"{log_path.resolve()}"
    )

    logger.info(
        f"Python:           "
        f"{sys.version.split()[0]}"
    )

    logger.info(
        f"PyTorch:          "
        f"{torch.__version__}"
    )

    logger.info(
        f"scikit-learn:     "
        f"{sklearn.__version__}"
    )

    logger.info(
        f"C:                "
        f"{args.C}"
    )

    logger.info(
        "Penalty:          L2"
    )

    logger.info(
        "Class weighting:  balanced"
    )

    logger.info(
        "Layer selection:  validation AUROC"
    )

    logger.info(
        "Threshold select: validation balanced accuracy"
    )

    logger.info(
        f"Bootstrap:        "
        f"{args.bootstrap} stratified resamples"
    )

    logger.info(
        f"Seed:             "
        f"{args.seed}"
    )

    logger.info(
        "Score semantics:  probe-derived insufficiency score, "
        "not calibrated probability"
    )

    logger.info("")
    logger.info(
        "Selected models:"
    )

    for model_name in (
        selected_models
    ):

        logger.info(
            f"  - {model_name}"
        )

    # ========================================================
    # Save global run config
    # ========================================================

    run_config = {
        "timestamp":
            datetime.now().isoformat(),

        "models":
            selected_models,

        "activation_root":
            str(
                activation_root.resolve()
            ),

        "output_root":
            str(
                output_root.resolve()
            ),

        "probe": {
            "type":
                "logistic_regression",

            "penalty":
                "l2",

            "C":
                float(
                    args.C
                ),

            "class_weight":
                "balanced",

            "solver":
                "liblinear",

            "max_iter":
                int(
                    args.max_iter
                ),

            "seed":
                int(
                    args.seed
                ),
        },

        "selection": {
            "layer":
                "maximum HealthBench validation AUROC",

            "threshold":
                (
                    "maximum HealthBench validation "
                    "balanced accuracy"
                ),
        },

        "bootstrap": {
            "resamples":
                int(
                    args.bootstrap
                ),

            "type":
                "stratified",

            "confidence_level":
                0.95,
        },

        "labels": {
            "0":
                "evidence sufficient",

            "1":
                "evidence insufficient",
        },

        "clindet_mapping": {
            "Complete":
                0,

            "Incomplete_Determinable":
                0,

            "Incomplete_Undeterminable":
                1,
        },
    }

    with open(
        output_root
        / "run_config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            run_config,
            f,
            indent=2,
        )

    # ========================================================
    # Run models sequentially
    # ========================================================

    summary_rows = []

    for model_name in (
        selected_models
    ):

        model_rows = (
            run_model(
                model_name=model_name,

                activation_root=
                    activation_root,

                output_root=
                    output_root,

                C=args.C,

                seed=args.seed,

                max_iter=
                    args.max_iter,

                n_bootstrap=
                    args.bootstrap,

                logger=logger,
            )
        )

        summary_rows.extend(
            model_rows
        )

        # Save after every model so partial progress survives
        # if a later model unexpectedly fails.
        pd.DataFrame(
            summary_rows
        ).to_csv(
            output_root
            / "summary.csv",
            index=False,
        )

    # ========================================================
    # Final summary
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        output_root
        / "summary.csv",
        index=False,
    )

    logger.info("")
    logger.info(
        "=" * 78
    )

    logger.info(
        "ALL PROBE EXPERIMENTS COMPLETE"
    )

    logger.info(
        "=" * 78
    )

    logger.info("")
    logger.info(
        "Main summary:"
    )

    logger.info(
        str(
            (
                output_root
                / "summary.csv"
            ).resolve()
        )
    )

    logger.info("")
    logger.info(
        "Log:"
    )

    logger.info(
        str(
            log_path.resolve()
        )
    )

    logger.info("")
    logger.info(
        summary_df.to_string(
            index=False
        )
    )

    logger.info("")
    logger.info(
        f"End time: {datetime.now()}"
    )


if __name__ == "__main__":
    main()