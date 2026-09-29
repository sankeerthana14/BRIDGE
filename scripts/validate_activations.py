# -*- coding: utf-8 -*-

"""
BRIDGE Revision - Activation Validation
=======================================

This script performs two checks:

1. Full activation-cache validation
   - expected file counts
   - tensor shapes
   - finite values
   - metadata consistency
   - expected total of 4485 activation files

2. UltraMedical context-length audit
   - tokenizer.model_max_length
   - config.max_position_embeddings
   - rope_scaling
   - maximum token count actually observed
   - examples exceeding tokenizer/model context limits

All console output is also written to:

    logs/activation_validation_YYYYMMDD_HHMMSS.log

Usage:

    python scripts/validate_activations.py
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_OUTPUT_ROOT = Path("cache/activations")
DEFAULT_MODEL_ROOT = Path("models")
DEFAULT_LOG_DIR = Path("logs")


# ============================================================
# EXPECTED MODEL ARCHITECTURES
# ============================================================

MODELS = {
    "biomistral": {
        "shape": (32, 4096),
    },

    "openbiollm": {
        "shape": (32, 4096),
    },

    "ultramedical": {
        "shape": (32, 4096),
    },

    "medgemma": {
        "shape": (34, 2560),
    },

    "lingshu": {
        "shape": (28, 3584),
    },
}


# ============================================================
# EXPECTED DATASET COUNTS
# ============================================================

EXPECTED = {
    ("healthbench", "train"): 492,
    ("healthbench", "validation"): 123,
    ("healthbench", "test"): 188,
    ("clindet", "external_test"): 94,
}


EXPECTED_PER_MODEL = sum(
    EXPECTED.values()
)

EXPECTED_TOTAL = (
    EXPECTED_PER_MODEL
    * len(MODELS)
)


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
        / f"activation_validation_{timestamp}.log"
    )

    logger = logging.getLogger(
        "bridge_validation"
    )

    logger.setLevel(
        logging.INFO
    )

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(message)s"
    )

    # Console
    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    # File
    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
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
# ACTIVATION VALIDATION
# ============================================================

def validate_activations(
    output_root,
    logger,
):
    """
    Validate every saved activation file.
    """

    logger.info(
        "=" * 70
    )

    logger.info(
        "TEST 1 - ACTIVATION CACHE VALIDATION"
    )

    logger.info(
        "=" * 70
    )

    logger.info("")

    all_ok = True

    total_files = 0

    model_token_counts = {
        model: []
        for model in MODELS
    }

    model_example_info = {
        model: []
        for model in MODELS
    }

    for (
        model,
        model_info,
    ) in MODELS.items():

        expected_shape = (
            model_info["shape"]
        )

        logger.info(
            "=" * 70
        )

        logger.info(
            model
        )

        logger.info(
            "=" * 70
        )

        model_total = 0

        for (
            dataset,
            split,
        ), expected_count in EXPECTED.items():

            directory = (
                output_root
                / model
                / dataset
                / split
            )

            if directory.exists():

                files = sorted(
                    directory.glob(
                        "*.pt"
                    )
                )

            else:

                files = []

            actual_count = len(
                files
            )

            model_total += (
                actual_count
            )

            total_files += (
                actual_count
            )

            count_ok = (
                actual_count
                == expected_count
            )

            status = (
                "OK"
                if count_ok
                else "MISMATCH"
            )

            logger.info(
                f"{dataset:12s} "
                f"{split:15s} "
                f"actual={actual_count:4d} "
                f"expected={expected_count:4d} "
                f"{status}"
            )

            if not count_ok:

                all_ok = False

                continue

            # =================================================
            # Validate every file
            # =================================================

            for path in files:

                try:

                    obj = torch.load(
                        path,
                        map_location="cpu",
                        weights_only=False,
                    )

                except Exception as e:

                    logger.info(
                        "  LOAD ERROR: "
                        f"{path.name}: "
                        f"{e}"
                    )

                    all_ok = False

                    continue

                # ---------------------------------------------
                # Required keys
                # ---------------------------------------------

                required_keys = {
                    "activations",
                    "dataset",
                    "split",
                    "model_name",
                    "label",
                    "example_id",
                    "token_count",
                }

                missing = (
                    required_keys
                    - set(obj.keys())
                )

                if missing:

                    logger.info(
                        "  MISSING KEYS: "
                        f"{path.name}: "
                        f"{sorted(missing)}"
                    )

                    all_ok = False

                    continue

                x = obj[
                    "activations"
                ]

                # ---------------------------------------------
                # Tensor shape
                # ---------------------------------------------

                if (
                    tuple(x.shape)
                    != expected_shape
                ):

                    logger.info(
                        "  BAD SHAPE: "
                        f"{path.name}: "
                        f"{tuple(x.shape)} "
                        f"expected "
                        f"{expected_shape}"
                    )

                    all_ok = False

                # ---------------------------------------------
                # Numeric validity
                # ---------------------------------------------

                if not torch.isfinite(
                    x
                ).all():

                    logger.info(
                        "  NON-FINITE: "
                        f"{path.name}"
                    )

                    all_ok = False

                # ---------------------------------------------
                # Metadata
                # ---------------------------------------------

                if (
                    obj["dataset"]
                    != dataset
                ):

                    logger.info(
                        "  DATASET MISMATCH: "
                        f"{path.name}"
                    )

                    all_ok = False

                if (
                    obj["split"]
                    != split
                ):

                    logger.info(
                        "  SPLIT MISMATCH: "
                        f"{path.name}"
                    )

                    all_ok = False

                if (
                    obj["model_name"]
                    != model
                ):

                    logger.info(
                        "  MODEL MISMATCH: "
                        f"{path.name}"
                    )

                    all_ok = False

                # ---------------------------------------------
                # Token counts
                # ---------------------------------------------

                token_count = int(
                    obj[
                        "token_count"
                    ]
                )

                model_token_counts[
                    model
                ].append(
                    token_count
                )

                model_example_info[
                    model
                ].append(
                    {
                        "example_id":
                            obj[
                                "example_id"
                            ],

                        "dataset":
                            dataset,

                        "split":
                            split,

                        "token_count":
                            token_count,
                    }
                )

        logger.info("")

        logger.info(
            f"Model total: "
            f"{model_total}"
        )

        logger.info(
            f"Expected:    "
            f"{EXPECTED_PER_MODEL}"
        )

        if (
            model_total
            != EXPECTED_PER_MODEL
        ):

            all_ok = False

        if model_token_counts[
            model
        ]:

            logger.info(
                "Token count range: "
                f"{min(model_token_counts[model])} "
                "to "
                f"{max(model_token_counts[model])}"
            )

        logger.info("")

    # ========================================================
    # Global total
    # ========================================================

    logger.info(
        "=" * 70
    )

    logger.info(
        "GLOBAL FILE COUNT"
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        f"Actual:   {total_files}"
    )

    logger.info(
        f"Expected: {EXPECTED_TOTAL}"
    )

    if (
        total_files
        != EXPECTED_TOTAL
    ):

        all_ok = False

    logger.info("")

    if all_ok:

        logger.info(
            "TEST 1 RESULT: PASS"
        )

    else:

        logger.info(
            "TEST 1 RESULT: FAIL"
        )

    logger.info("")

    return (
        all_ok,
        model_token_counts,
        model_example_info,
    )


# ============================================================
# ULTRAMEDICAL CONTEXT-LENGTH AUDIT
# ============================================================

def validate_ultramedical_context(
    model_root,
    model_token_counts,
    model_example_info,
    logger,
):
    """
    Investigate the warning:

        token sequence length 1094 > 1024

    We compare actual saved prompt lengths against both:

        tokenizer.model_max_length
        model config max_position_embeddings
    """

    logger.info(
        "=" * 70
    )

    logger.info(
        "TEST 2 - ULTRAMEDICAL CONTEXT LENGTH AUDIT"
    )

    logger.info(
        "=" * 70
    )

    logger.info("")

    model_path = (
        model_root
        / "ultramedical"
    )

    if not model_path.exists():

        logger.info(
            "ERROR: UltraMedical model "
            f"directory not found: "
            f"{model_path}"
        )

        return False

    # ========================================================
    # Load tokenizer and config
    # ========================================================

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )
    )

    config = (
        AutoConfig
        .from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )
    )

    tokenizer_limit = getattr(
        tokenizer,
        "model_max_length",
        None,
    )

    config_limit = getattr(
        config,
        "max_position_embeddings",
        None,
    )

    rope_scaling = getattr(
        config,
        "rope_scaling",
        None,
    )

    logger.info(
        "Tokenizer model_max_length:"
    )

    logger.info(
        str(
            tokenizer_limit
        )
    )

    logger.info("")

    logger.info(
        "Model config max_position_embeddings:"
    )

    logger.info(
        str(
            config_limit
        )
    )

    logger.info("")

    logger.info(
        "Model config rope_scaling:"
    )

    logger.info(
        str(
            rope_scaling
        )
    )

    logger.info("")

    token_counts = (
        model_token_counts.get(
            "ultramedical",
            [],
        )
    )

    examples = (
        model_example_info.get(
            "ultramedical",
            [],
        )
    )

    if not token_counts:

        logger.info(
            "ERROR: No UltraMedical "
            "activation token counts found."
        )

        return False

    minimum = min(
        token_counts
    )

    maximum = max(
        token_counts
    )

    mean = (
        sum(token_counts)
        / len(token_counts)
    )

    logger.info(
        "Observed token counts:"
    )

    logger.info(
        f"  Examples: {len(token_counts)}"
    )

    logger.info(
        f"  Minimum:  {minimum}"
    )

    logger.info(
        f"  Maximum:  {maximum}"
    )

    logger.info(
        f"  Mean:     {mean:.2f}"
    )

    logger.info("")

    # ========================================================
    # Compare against tokenizer metadata
    # ========================================================

    tokenizer_over = []

    if (
        tokenizer_limit
        is not None
        and
        isinstance(
            tokenizer_limit,
            int,
        )
        and
        tokenizer_limit < 10**12
    ):

        tokenizer_over = [
            item
            for item
            in examples
            if (
                item[
                    "token_count"
                ]
                > tokenizer_limit
            )
        ]

        logger.info(
            "Examples above tokenizer "
            f"model_max_length "
            f"({tokenizer_limit}): "
            f"{len(tokenizer_over)}"
        )

        for item in (
            sorted(
                tokenizer_over,
                key=lambda x: (
                    x[
                        "token_count"
                    ]
                ),
                reverse=True,
            )[:20]
        ):

            logger.info(
                "  "
                f"{item['example_id']} "
                f"dataset={item['dataset']} "
                f"split={item['split']} "
                f"tokens={item['token_count']}"
            )

    else:

        logger.info(
            "Tokenizer model_max_length "
            "does not provide a practical limit."
        )

    logger.info("")

    # ========================================================
    # Compare against actual model architecture limit
    # ========================================================

    config_over = []

    if (
        config_limit
        is not None
        and
        isinstance(
            config_limit,
            int,
        )
    ):

        config_over = [
            item
            for item
            in examples
            if (
                item[
                    "token_count"
                ]
                > config_limit
            )
        ]

        logger.info(
            "Examples above model config "
            "max_position_embeddings "
            f"({config_limit}): "
            f"{len(config_over)}"
        )

        for item in (
            sorted(
                config_over,
                key=lambda x: (
                    x[
                        "token_count"
                    ]
                ),
                reverse=True,
            )[:20]
        ):

            logger.info(
                "  "
                f"{item['example_id']} "
                f"dataset={item['dataset']} "
                f"split={item['split']} "
                f"tokens={item['token_count']}"
            )

    else:

        logger.info(
            "Model config does not provide "
            "max_position_embeddings."
        )

    logger.info("")

    # ========================================================
    # Interpretation
    # ========================================================

    if config_limit is None:

        logger.info(
            "TEST 2 RESULT: REVIEW REQUIRED"
        )

        logger.info(
            "Reason: model configuration does not "
            "provide an explicit architecture context limit."
        )

        return False

    if len(
        config_over
    ) > 0:

        logger.info(
            "TEST 2 RESULT: FAIL"
        )

        logger.info(
            "Reason: one or more prompts exceed "
            "UltraMedical's configured positional limit."
        )

        logger.info(
            "These examples need a documented "
            "truncation/exclusion policy before probe training."
        )

        return False

    # If tokenizer says 1024 but config supports more,
    # that warning is likely tokenizer metadata rather than
    # an actual model limit.
    if (
        tokenizer_limit
        is not None
        and
        tokenizer_limit
        < config_limit
        and
        len(tokenizer_over)
        > 0
    ):

        logger.info(
            "TEST 2 RESULT: PASS WITH NOTE"
        )

        logger.info(
            "Some prompts exceed tokenizer.model_max_length, "
            "but none exceed model.config."
            "max_position_embeddings."
        )

        logger.info(
            "The earlier warning therefore appears to be "
            "caused by conservative/stale tokenizer metadata "
            "rather than the model architecture limit."
        )

        return True

    logger.info(
        "TEST 2 RESULT: PASS"
    )

    logger.info(
        "No extracted UltraMedical prompt exceeds "
        "the configured model context limit."
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Validate BRIDGE activation cache and "
            "audit UltraMedical context length."
        )
    )

    parser.add_argument(
        "--output-root",
        default=str(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    parser.add_argument(
        "--model-root",
        default=str(
            DEFAULT_MODEL_ROOT
        ),
    )

    parser.add_argument(
        "--log-dir",
        default=str(
            DEFAULT_LOG_DIR
        ),
    )

    args = parser.parse_args()

    output_root = Path(
        args.output_root
    )

    model_root = Path(
        args.model_root
    )

    log_dir = Path(
        args.log_dir
    )

    logger, log_path = (
        setup_logging(
            log_dir
        )
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "BRIDGE ACTIVATION VALIDATION"
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        f"Time:        {datetime.now()}"
    )

    logger.info(
        f"Output root: {output_root.resolve()}"
    )

    logger.info(
        f"Model root:  {model_root.resolve()}"
    )

    logger.info(
        f"Log file:    {log_path.resolve()}"
    )

    logger.info("")

    # ========================================================
    # Test 1
    # ========================================================

    (
        activation_ok,
        model_token_counts,
        model_example_info,
    ) = validate_activations(
        output_root=output_root,
        logger=logger,
    )

    # ========================================================
    # Test 2
    # ========================================================

    context_ok = (
        validate_ultramedical_context(
            model_root=model_root,

            model_token_counts=
                model_token_counts,

            model_example_info=
                model_example_info,

            logger=logger,
        )
    )

    # ========================================================
    # Final summary
    # ========================================================

    logger.info("")
    logger.info(
        "=" * 70
    )

    logger.info(
        "FINAL SUMMARY"
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "Activation cache validation: "
        + (
            "PASS"
            if activation_ok
            else "FAIL"
        )
    )

    logger.info(
        "UltraMedical context audit:   "
        + (
            "PASS"
            if context_ok
            else "FAIL"
        )
    )

    logger.info("")

    logger.info(
        "Log saved to:"
    )

    logger.info(
        str(
            log_path.resolve()
        )
    )

    logger.info("")

    if (
        activation_ok
        and
        context_ok
    ):

        logger.info(
            "ALL VALIDATION CHECKS PASSED"
        )

        raise SystemExit(
            0
        )

    logger.info(
        "ONE OR MORE VALIDATION CHECKS FAILED"
    )

    raise SystemExit(
        1
    )


if __name__ == "__main__":
    main()