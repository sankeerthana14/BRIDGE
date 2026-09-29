# -*- coding: utf-8 -*-

"""
BRIDGE Revision - Activation Extraction
=======================================

Extract last-prompt-token hidden states from every transformer
language layer for:

TARGET MODELS
-------------
1. BioMistral
2. OpenBioLLM
3. UltraMedical
4. MedGemma 1.5
5. Lingshu-7B

DATASETS
--------
1. HealthBench
   - train
   - validation
   - test (HealthBench Hard)

2. ClinDet-Bench
   - external_test only

Each example is saved independently as a .pt file.

Saved activation tensor shape:

    [num_layers, hidden_size]

Examples:

    BioMistral   -> [32, 4096]
    OpenBioLLM   -> [32, 4096]
    UltraMedical -> [32, 4096]
    MedGemma     -> [34, 2560]
    Lingshu      -> [28, 3584]

Smoke test:

python scripts/extract_activations.py \
    --models all \
    --datasets all \
    --output-root cache/activations_smoke \
    --max-examples-per-split 1

Full run:

python scripts/extract_activations.py \
    --models all \
    --datasets all \
    --output-root cache/activations
"""

import argparse
import gc
import hashlib
import inspect
import json
import math
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_SPECS = {
    # Original manuscript models
    "biomistral": {
        "path": "models/biomistral",
        "type": "causal",
    },

    "openbiollm": {
        "path": "models/openbiollm",
        "type": "causal",
    },

    "ultramedical": {
        "path": "models/ultramedical",
        "type": "causal",
    },

    # New reviewer-requested models
    "medgemma": {
        "path": "models/medgemma",
        "type": "multimodal_text",
    },

    "lingshu": {
        "path": "models/lingshu",
        "type": "multimodal_text",
    },
}


HEALTHBENCH_MANIFEST = (
    "data/manifests/"
    "healthbench_evidence_sufficiency_v1.csv"
)

CLINDET_ROOT = "data/clindet"

DEFAULT_OUTPUT_ROOT = "cache/activations"


# ============================================================
# BASIC HELPERS
# ============================================================

def log(message=""):
    print(message, flush=True)


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def safe_filename(text):
    return re.sub(
        r"[^A-Za-z0-9_.-]",
        "_",
        str(text),
    )


def is_missing(value):
    """
    Robust missing-value check that does not break on lists/dicts.
    """

    if value is None:
        return True

    if isinstance(value, float):
        return math.isnan(value)

    return False


# ============================================================
# HEALTHBENCH LOADING
# ============================================================

def parse_healthbench_messages(prompt_value):
    """
    Convert the prompt stored in the frozen manifest into
    HuggingFace-style chat messages.

    Expected form:

        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]

    If parsing fails, treat the entire value as one user message.
    """

    if is_missing(prompt_value):
        raise ValueError(
            "HealthBench prompt is missing."
        )

    parsed = None

    if isinstance(prompt_value, list):
        parsed = prompt_value

    elif isinstance(prompt_value, str):
        try:
            parsed = json.loads(
                prompt_value
            )
        except Exception:
            parsed = None

    if isinstance(parsed, list):
        messages = []

        for message in parsed:

            if not isinstance(
                message,
                dict,
            ):
                continue

            role = message.get(
                "role",
                "user",
            )

            content = message.get(
                "content",
                "",
            )

            if not isinstance(
                content,
                str,
            ):
                content = json.dumps(
                    content,
                    ensure_ascii=False,
                )

            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        if messages:
            return messages

    return [
        {
            "role": "user",
            "content": str(prompt_value),
        }
    ]


def load_healthbench_examples():
    """
    Load only included examples from the frozen HealthBench
    manifest.

    Expected split sizes:

        train        492
        validation   123
        test         188
    """

    manifest_path = Path(
        HEALTHBENCH_MANIFEST
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "HealthBench manifest not found: "
            f"{manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    required_columns = {
        "example_id",
        "included",
        "split",
        "binary_label",
        "prompt",
    }

    missing_columns = (
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "HealthBench manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    included_mask = (
        manifest["included"]
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

    manifest = manifest[
        included_mask
    ].copy()

    examples = []

    for _, row in manifest.iterrows():

        split = str(
            row["split"]
        )

        if split not in {
            "train",
            "validation",
            "test",
        }:
            continue

        messages = (
            parse_healthbench_messages(
                row["prompt"]
            )
        )

        examples.append(
            {
                "example_id":
                    str(
                        row["example_id"]
                    ),

                "dataset":
                    "healthbench",

                "split":
                    split,

                "label":
                    int(
                        row["binary_label"]
                    ),

                "messages":
                    messages,

                "source_metadata": {
                    "prompt_hash":
                        row.get(
                            "prompt_hash",
                            None,
                        ),

                    "physician_categories":
                        row.get(
                            "physician_agreed_categories",
                            None,
                        ),
                },
            }
        )

    return examples


# ============================================================
# CLINDET LOADING
# ============================================================

def find_clindet_file():
    """
    Prefer the official ClinDet base result file because it
    contains the clinical_decision_task prompt.

    Fall back to the raw benchmark file if needed.
    """

    root = Path(
        CLINDET_ROOT
    )

    if not root.exists():
        raise FileNotFoundError(
            "ClinDet directory not found: "
            f"{root}"
        )

    preferred = list(
        root.rglob(
            "Clinical_Decision_Task_Result_base.xlsx"
        )
    )

    if preferred:
        return (
            preferred[0],
            "results",
        )

    raw = list(
        root.rglob(
            "Clinical_Decision_Task.xlsx"
        )
    )

    if raw:
        return (
            raw[0],
            "raw",
        )

    raise FileNotFoundError(
        "Could not find ClinDet Clinical Decision "
        f"Task files under {CLINDET_ROOT}"
    )


def make_clindet_fallback_prompt(row):
    """
    Construct a prompt only if the official ClinDet prompt is
    unavailable.

    Evidence sufficiency is relative to a clinical criterion,
    so both criterion and case are included.
    """

    score_name = str(
        row.get(
            "score_name",
            "",
        )
    )

    criterion = str(
        row.get(
            "criterion",
            "",
        )
    )

    case = str(
        row.get(
            "case_presentation",
            "",
        )
    )

    return (
        f"Clinical scoring system: {score_name}\n\n"
        f"Clinical criterion to assess:\n"
        f"{criterion}\n\n"
        f"Clinical case:\n"
        f"{case}\n\n"
        "Using only the information provided, determine "
        "whether the clinical criterion can be assessed "
        "and whether it is met."
    )


def clindet_binary_label(condition):
    """
    External evidence-sufficiency mapping.

    0 = sufficient for the requested clinical judgment
    1 = insufficient for the requested clinical judgment

    Complete:
        0

    Incomplete_Determinable:
        0

    Incomplete_Undeterminable:
        1

    Incomplete_Determinable is intentionally treated as a
    hard negative: information is missing, but the target
    clinical judgment remains determinable.
    """

    condition = str(
        condition
    ).strip()

    if condition in {
        "Complete",
        "Incomplete_Determinable",
    }:
        return 0

    if (
        condition
        == "Incomplete_Undeterminable"
    ):
        return 1

    raise ValueError(
        "Unknown ClinDet information condition: "
        f"{condition}"
    )


def load_clindet_examples():
    """
    Load ClinDet examples.

    ClinDet is external-test only.

    It is never used for:
        - probe training
        - layer selection
        - threshold selection
    """

    path, source_type = (
        find_clindet_file()
    )

    log(
        f"ClinDet source: {path}"
    )

    df = pd.read_excel(
        path
    )

    if (
        "information_condition"
        not in df.columns
    ):
        raise ValueError(
            "ClinDet file does not contain "
            "'information_condition'."
        )

    examples = []

    for idx, row in df.iterrows():

        condition = row[
            "information_condition"
        ]

        label = (
            clindet_binary_label(
                condition
            )
        )

        if (
            source_type == "results"
            and
            "clinical_decision_task"
            in df.columns
            and
            not is_missing(
                row[
                    "clinical_decision_task"
                ]
            )
        ):
            prompt = str(
                row[
                    "clinical_decision_task"
                ]
            )

        else:
            prompt = (
                make_clindet_fallback_prompt(
                    row
                )
            )

        identity = (
            f"{row.get('score_name', '')}|"
            f"{row.get('criterion', '')}|"
            f"{row.get('case_presentation', '')}|"
            f"{condition}"
        )

        digest = (
            sha256_text(
                identity
            )[:12]
        )

        example_id = (
            f"clindet_"
            f"{idx:03d}_"
            f"{digest}"
        )

        examples.append(
            {
                "example_id":
                    example_id,

                "dataset":
                    "clindet",

                "split":
                    "external_test",

                "label":
                    label,

                "messages": [
                    {
                        "role":
                            "user",

                        "content":
                            prompt,
                    }
                ],

                "source_metadata": {
                    "information_condition":
                        str(
                            condition
                        ),

                    "score_name":
                        str(
                            row.get(
                                "score_name",
                                "",
                            )
                        ),

                    "criterion":
                        str(
                            row.get(
                                "criterion",
                                "",
                            )
                        ),

                    "ground_truth_label":
                        str(
                            row.get(
                                "ground_truth_label",
                                row.get(
                                    "classification_ground_truth",
                                    "",
                                ),
                            )
                        ),
                },
            }
        )

    return examples


# ============================================================
# DATASET ROUTER
# ============================================================

def load_dataset_examples(name):

    if name == "healthbench":
        return (
            load_healthbench_examples()
        )

    if name == "clindet":
        return (
            load_clindet_examples()
        )

    raise ValueError(
        f"Unknown dataset: {name}"
    )


# ============================================================
# MODEL CONFIG HELPERS
# ============================================================

def choose_dtype():
    """
    Use BF16 when supported.

    RTX 6000 Ada supports BF16.
    """

    if (
        torch.cuda.is_available()
        and
        torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16

    return torch.float16


def get_config_value(
    config,
    key,
):
    """
    Search common text-model config locations.

    Multimodal models may store the language config inside
    text_config or language_config.
    """

    candidates = [
        config,

        getattr(
            config,
            "text_config",
            None,
        ),

        getattr(
            config,
            "language_config",
            None,
        ),

        getattr(
            config,
            "llm_config",
            None,
        ),
    ]

    for candidate in candidates:

        if candidate is None:
            continue

        value = getattr(
            candidate,
            key,
            None,
        )

        if value is not None:
            return value

    return None


# ============================================================
# MULTIMODAL MODEL LOADING
# ============================================================

def load_multimodal_model(path):
    """
    Load MedGemma or Lingshu.
    """

    errors = []

    try:
        from transformers import (
            AutoModelForImageTextToText
        )

        model = (
            AutoModelForImageTextToText
            .from_pretrained(
                path,
                dtype=choose_dtype(),
                device_map="auto",
                trust_remote_code=True,
            )
        )

        return model

    except Exception as e:
        errors.append(
            (
                "AutoModelForImageTextToText",
                str(e),
            )
        )

    try:
        from transformers import (
            Qwen2_5_VLForConditionalGeneration
        )

        model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                path,
                dtype=choose_dtype(),
                device_map="auto",
                trust_remote_code=True,
            )
        )

        return model

    except Exception as e:
        errors.append(
            (
                "Qwen2_5_VLForConditionalGeneration",
                str(e),
            )
        )

    error_text = "\n".join(
        f"{name}: {error}"
        for name, error
        in errors
    )

    raise RuntimeError(
        "Unable to load multimodal model.\n"
        + error_text
    )


def load_model_bundle(model_name):
    """
    Load exactly one model at a time.
    """

    if model_name not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    spec = MODEL_SPECS[
        model_name
    ]

    path = Path(
        spec["path"]
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {path}"
        )

    log()
    log("=" * 70)
    log(
        f"LOADING MODEL: {model_name}"
    )
    log("=" * 70)
    log(
        f"Path: {path}"
    )

    if (
        spec["type"]
        == "causal"
    ):

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                str(path),
                trust_remote_code=True,
            )
        )

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                str(path),
                dtype=choose_dtype(),
                device_map="auto",
                trust_remote_code=True,
            )
        )

        formatter = tokenizer

    else:

        # Lazy import so text-only models do not require
        # AutoProcessor until needed.
        from transformers import (
            AutoProcessor
        )

        processor = (
            AutoProcessor
            .from_pretrained(
                str(path),
                trust_remote_code=True,
            )
        )

        model = (
            load_multimodal_model(
                str(path)
            )
        )

        formatter = processor

    model.eval()

    expected_layers = (
        get_config_value(
            model.config,
            "num_hidden_layers",
        )
    )

    hidden_size = (
        get_config_value(
            model.config,
            "hidden_size",
        )
    )

    log(
        "Expected language layers: "
        f"{expected_layers}"
    )

    log(
        "Expected hidden size: "
        f"{hidden_size}"
    )

    return {
        "name":
            model_name,

        "spec":
            spec,

        "model":
            model,

        "formatter":
            formatter,

        "expected_layers":
            expected_layers,

        "hidden_size":
            hidden_size,
    }


# ============================================================
# INPUT NORMALIZATION
# ============================================================

def normalize_text_inputs(encoded):
    """
    Normalize tokenizer output.

    Some tokenizers return a raw Tensor.
    Others return BatchEncoding or dict-like objects.
    """

    if torch.is_tensor(
        encoded
    ):

        input_ids = encoded

        if input_ids.ndim == 1:
            input_ids = (
                input_ids.unsqueeze(0)
            )

        return {
            "input_ids":
                input_ids,

            "attention_mask":
                torch.ones_like(
                    input_ids
                ),
        }

    if hasattr(
        encoded,
        "items",
    ):

        result = {
            key: value
            for key, value
            in encoded.items()
        }

        if (
            "input_ids"
            not in result
        ):
            raise RuntimeError(
                "Tokenizer output does not "
                "contain input_ids."
            )

        input_ids = result[
            "input_ids"
        ]

        if not torch.is_tensor(
            input_ids
        ):
            raise TypeError(
                "encoded['input_ids'] is not "
                f"a Tensor: {type(input_ids)}"
            )

        if input_ids.ndim == 1:

            input_ids = (
                input_ids.unsqueeze(0)
            )

            result[
                "input_ids"
            ] = input_ids

        if (
            "attention_mask"
            not in result
            or
            result[
                "attention_mask"
            ] is None
        ):
            result[
                "attention_mask"
            ] = torch.ones_like(
                input_ids
            )

        return result

    raise TypeError(
        "Unsupported tokenizer output type: "
        f"{type(encoded)}"
    )


# ============================================================
# CHAT FORMATTING
# ============================================================

def fallback_plain_text(messages):
    """
    Fallback for tokenizers without an official chat template.
    """

    parts = []

    for message in messages:

        role = message.get(
            "role",
            "user",
        )

        content = message.get(
            "content",
            "",
        )

        parts.append(
            f"{role}: {content}"
        )

    parts.append(
        "assistant:"
    )

    return "\n".join(
        parts
    )


def convert_messages_for_multimodal(messages):
    """
    Multimodal processors usually expect content entries like:

        {
            "type": "text",
            "text": "..."
        }

    even for text-only inputs.
    """

    converted = []

    for message in messages:

        converted.append(
            {
                "role":
                    message.get(
                        "role",
                        "user",
                    ),

                "content": [
                    {
                        "type":
                            "text",

                        "text":
                            str(
                                message.get(
                                    "content",
                                    "",
                                )
                            ),
                    }
                ],
            }
        )

    return converted


def build_inputs(
    bundle,
    messages,
):
    """
    Format one conversation according to the target model's
    tokenizer or processor.
    """

    spec = bundle[
        "spec"
    ]

    formatter = bundle[
        "formatter"
    ]

    # ========================================================
    # Standard causal LMs
    # ========================================================

    if (
        spec["type"]
        == "causal"
    ):

        chat_template = getattr(
            formatter,
            "chat_template",
            None,
        )

        if chat_template:

            try:
                encoded = (
                    formatter
                    .apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                encoded = (
                    formatter
                    .apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                )

            return (
                normalize_text_inputs(
                    encoded
                )
            )

        text = (
            fallback_plain_text(
                messages
            )
        )

        encoded = formatter(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )

        return (
            normalize_text_inputs(
                encoded
            )
        )

    # ========================================================
    # Multimodal models used in text-only mode
    # ========================================================

    multimodal_messages = (
        convert_messages_for_multimodal(
            messages
        )
    )

    try:
        encoded = (
            formatter
            .apply_chat_template(
                multimodal_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        )

        return (
            normalize_text_inputs(
                encoded
            )
        )

    except Exception:

        text = (
            formatter
            .apply_chat_template(
                multimodal_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        )

        encoded = formatter(
            text=[
                text
            ],
            return_tensors="pt",
        )

        return (
            normalize_text_inputs(
                encoded
            )
        )


# ============================================================
# DEVICE HANDLING
# ============================================================

def get_input_device(model):
    """
    Find the device containing the model's token embeddings.
    """

    try:

        embeddings = (
            model
            .get_input_embeddings()
        )

        return (
            embeddings
            .weight
            .device
        )

    except Exception:
        pass

    try:
        return model.device

    except Exception:
        pass

    return next(
        model.parameters()
    ).device


def move_inputs_to_device(
    inputs,
    device,
):
    """
    Move tensor values only.
    """

    moved = {}

    for key, value in (
        inputs.items()
    ):

        if torch.is_tensor(
            value
        ):
            moved[
                key
            ] = value.to(
                device
            )

        else:
            moved[
                key
            ] = value

    return moved


# ============================================================
# HIDDEN STATE HELPERS
# ============================================================

def find_hidden_states(outputs):
    """
    Find language hidden states in common Transformers output
    layouts.
    """

    hidden_states = getattr(
        outputs,
        "hidden_states",
        None,
    )

    if (
        hidden_states
        is not None
    ):
        return hidden_states

    for name in [
        "language_model_output",
        "model_output",
        "text_model_output",
    ]:

        nested = getattr(
            outputs,
            name,
            None,
        )

        if nested is None:
            continue

        hidden_states = getattr(
            nested,
            "hidden_states",
            None,
        )

        if (
            hidden_states
            is not None
        ):
            return hidden_states

    raise RuntimeError(
        "Forward pass did not return hidden_states."
    )


def select_transformer_layers(
    hidden_states,
    expected_layers,
):
    """
    Resolve HuggingFace hidden-state indexing.

    Standard convention:

        hidden_states[0] = embedding output
        hidden_states[1] = transformer layer 0
        ...
        hidden_states[L] = transformer layer L-1

    Some custom models return only L transformer states.
    """

    n_states = len(
        hidden_states
    )

    if (
        expected_layers
        is not None
        and
        n_states
        == expected_layers + 1
    ):
        return (
            hidden_states[1:],
            1,
        )

    if (
        expected_layers
        is not None
        and
        n_states
        == expected_layers
    ):
        return (
            hidden_states,
            0,
        )

    raise RuntimeError(
        "Unexpected hidden-state count: "
        f"got {n_states}, "
        f"expected {expected_layers} "
        "or "
        f"{expected_layers + 1 if expected_layers is not None else '?'}."
    )


def hash_input_ids(
    input_ids,
):
    """
    Hash the exact model input token sequence.
    """

    values = (
        input_ids
        .detach()
        .cpu()
        .reshape(-1)
        .tolist()
    )

    payload = json.dumps(
        values,
        separators=(
            ",",
            ":",
        ),
    )

    return (
        sha256_text(
            payload
        )
    )


# ============================================================
# EXTRACT ONE EXAMPLE
# ============================================================

def extract_one_example(
    bundle,
    example,
):
    """
    Extract the final prompt-token hidden state from every
    language transformer layer.

    Saved activations are converted to FP32.

    This is intentional:
        - avoids BF16-to-FP16 overflow for MedGemma;
        - is convenient for sklearn/logistic regression;
        - preserves a large numerical range.

    Returns:
        activations with shape [num_layers, hidden_size]
    """

    model = bundle[
        "model"
    ]

    expected_layers = bundle[
        "expected_layers"
    ]

    inputs = build_inputs(
        bundle,
        example[
            "messages"
        ],
    )

    if (
        "input_ids"
        not in inputs
    ):
        raise RuntimeError(
            "Model formatter did not return input_ids."
        )

    # ========================================================
    # Metadata before moving inputs to GPU
    # ========================================================

    input_hash = (
        hash_input_ids(
            inputs[
                "input_ids"
            ]
        )
    )

    token_count = int(
        inputs[
            "input_ids"
        ].shape[-1]
    )

    # ========================================================
    # Find the final non-padding prompt token
    # ========================================================

    if (
        "attention_mask"
        in inputs
        and
        inputs[
            "attention_mask"
        ] is not None
    ):

        last_token_index = (
            int(
                inputs[
                    "attention_mask"
                ][0]
                .sum()
                .item()
            )
            - 1
        )

    else:
        last_token_index = (
            token_count
            - 1
        )

    if last_token_index < 0:
        raise RuntimeError(
            "Computed invalid last token index."
        )

    # ========================================================
    # Move model inputs to GPU
    # ========================================================

    device = (
        get_input_device(
            model
        )
    )

    inputs = (
        move_inputs_to_device(
            inputs,
            device,
        )
    )

    # ========================================================
    # Forward pass
    # ========================================================

    forward_kwargs = {
        **inputs,

        "output_hidden_states":
            True,

        "return_dict":
            True,

        "use_cache":
            False,
    }

    # Some Transformers models support avoiding unnecessary
    # full-sequence logits.
    try:

        signature = (
            inspect.signature(
                model.forward
            )
        )

        if (
            "logits_to_keep"
            in signature.parameters
        ):
            forward_kwargs[
                "logits_to_keep"
            ] = 1

    except Exception:
        pass

    with torch.inference_mode():

        outputs = model(
            **forward_kwargs
        )

    hidden_states = (
        find_hidden_states(
            outputs
        )
    )

    (
        layer_states,
        hidden_state_offset,
    ) = (
        select_transformer_layers(
            hidden_states,
            expected_layers,
        )
    )

    # ========================================================
    # Extract final-token vector from each layer
    # ========================================================

    activations = []

    for layer_idx, state in enumerate(
        layer_states
    ):

        if state.ndim != 3:
            raise RuntimeError(
                "Expected hidden state shape "
                "[batch, sequence, hidden], "
                f"got {tuple(state.shape)} "
                f"at layer {layer_idx}"
            )

        if (
            last_token_index
            >= state.shape[1]
        ):
            raise RuntimeError(
                "last_token_index exceeds hidden-state "
                "sequence length: "
                f"{last_token_index} >= "
                f"{state.shape[1]}"
            )

        vector = state[
            0,
            last_token_index,
            :
        ]

        # ----------------------------------------------------
        # IMPORTANT:
        # Check numerical validity BEFORE dtype conversion.
        # ----------------------------------------------------

        vector_cpu = (
            vector
            .detach()
            .cpu()
        )

        if not torch.isfinite(
            vector_cpu.float()
        ).all():

            num_nan = int(
                torch.isnan(
                    vector_cpu.float()
                ).sum()
                .item()
            )

            num_inf = int(
                torch.isinf(
                    vector_cpu.float()
                ).sum()
                .item()
            )

            raise RuntimeError(
                "Raw model hidden state contains "
                "NaN or Inf before dtype conversion. "
                f"layer={layer_idx}, "
                f"nan={num_nan}, "
                f"inf={num_inf}"
            )

        # ----------------------------------------------------
        # Store in FP32.
        #
        # DO NOT cast BF16 MedGemma activations directly
        # to FP16 because values outside the FP16 numerical
        # range can become Inf.
        # ----------------------------------------------------

        activations.append(
            vector_cpu.float()
        )

    activations = (
        torch.stack(
            activations,
            dim=0,
        )
    )

    # ========================================================
    # Sanity checks
    # ========================================================

    if activations.ndim != 2:
        raise RuntimeError(
            "Expected activation shape "
            "[num_layers, hidden_size], "
            f"got {tuple(activations.shape)}"
        )

    if (
        expected_layers
        is not None
        and
        activations.shape[0]
        != expected_layers
    ):
        raise RuntimeError(
            "Layer count mismatch: "
            f"got {activations.shape[0]}, "
            f"expected {expected_layers}"
        )

    if not torch.isfinite(
        activations
    ).all():
        raise RuntimeError(
            "Saved FP32 activation tensor "
            "contains NaN or Inf."
        )

    del outputs
    del hidden_states
    del layer_states

    return {
        "activations":
            activations,

        "token_count":
            token_count,

        "last_token_index":
            last_token_index,

        "input_hash":
            input_hash,

        "hidden_state_offset":
            hidden_state_offset,
    }


# ============================================================
# OUTPUT PATHS AND SAVING
# ============================================================

def get_destination_path(
    output_root,
    model_name,
    example,
):
    """
    Canonical file location for one activation example.
    """

    directory = (
        Path(output_root)
        / model_name
        / example["dataset"]
        / example["split"]
    )

    filename = (
        safe_filename(
            example[
                "example_id"
            ]
        )
        + ".pt"
    )

    return (
        directory
        / filename
    )


def save_example(
    output_root,
    model_name,
    example,
    result,
):
    """
    Save activation tensor and reproducibility metadata.
    """

    path = (
        get_destination_path(
            output_root,
            model_name,
            example,
        )
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    activations = result[
        "activations"
    ]

    torch.save(
        {
            # Identity
            "example_id":
                example[
                    "example_id"
                ],

            "dataset":
                example[
                    "dataset"
                ],

            "split":
                example[
                    "split"
                ],

            "label":
                int(
                    example[
                        "label"
                    ]
                ),

            "model_name":
                model_name,

            # Activations
            "activations":
                activations,

            "activation_dtype":
                str(
                    activations.dtype
                ),

            "num_layers":
                int(
                    activations.shape[0]
                ),

            "hidden_size":
                int(
                    activations.shape[1]
                ),

            # Tokenization metadata
            "token_count":
                int(
                    result[
                        "token_count"
                    ]
                ),

            "last_token_index":
                int(
                    result[
                        "last_token_index"
                    ]
                ),

            "input_hash":
                result[
                    "input_hash"
                ],

            "hidden_state_offset":
                int(
                    result[
                        "hidden_state_offset"
                    ]
                ),

            "token_position":
                (
                    "final non-padding token of the "
                    "model-formatted prompt immediately "
                    "before generation"
                ),

            "layer_indexing":
                (
                    "saved activations[l] corresponds "
                    "to transformer language layer l; "
                    "hidden_state_offset records whether "
                    "the raw hidden_states tuple included "
                    "an embedding-state entry"
                ),

            # Dataset metadata
            "source_metadata":
                example[
                    "source_metadata"
                ],
        },

        path,
    )

    return path


# ============================================================
# PROCESS ONE MODEL X DATASET
# ============================================================

def process_dataset(
    bundle,
    dataset_name,
    output_root,
    max_examples_per_split=None,
    overwrite=False,
):
    """
    Extract all examples for one model and one dataset.

    Returns:
        number of failed examples

    Any non-zero failure count is propagated to main(), which
    causes the entire process to exit with status code 1.
    """

    examples = (
        load_dataset_examples(
            dataset_name
        )
    )

    model_name = bundle[
        "name"
    ]

    log()
    log("=" * 70)
    log(
        f"{model_name.upper()} X "
        f"{dataset_name.upper()}"
    )
    log("=" * 70)

    # ========================================================
    # Smoke-test limiting, independently per split
    # ========================================================

    if (
        max_examples_per_split
        is not None
    ):

        limited = []

        splits = sorted(
            {
                example["split"]
                for example
                in examples
            }
        )

        for split in splits:

            subset = [
                example
                for example
                in examples
                if (
                    example[
                        "split"
                    ]
                    == split
                )
            ]

            limited.extend(
                subset[
                    :
                    max_examples_per_split
                ]
            )

        examples = limited

    # ========================================================
    # Print class summary
    # ========================================================

    summary = {}

    for example in examples:

        split = example[
            "split"
        ]

        if split not in summary:
            summary[
                split
            ] = {
                0: 0,
                1: 0,
            }

        summary[
            split
        ][
            int(
                example[
                    "label"
                ]
            )
        ] += 1

    log(
        "Examples to process: "
        f"{len(examples)}"
    )

    for (
        split,
        counts,
    ) in summary.items():

        log(
            f"  {split}: "
            f"class0={counts[0]}, "
            f"class1={counts[1]}"
        )

    # ========================================================
    # Extraction loop
    # ========================================================

    success = 0
    skipped = 0
    failures = []

    for example in tqdm(
        examples,
        desc=(
            f"{model_name}/"
            f"{dataset_name}"
        ),
    ):

        destination = (
            get_destination_path(
                output_root,
                model_name,
                example,
            )
        )

        # Resume support
        if (
            destination.exists()
            and
            not overwrite
        ):
            skipped += 1
            continue

        try:

            result = (
                extract_one_example(
                    bundle,
                    example,
                )
            )

            save_example(
                output_root,
                model_name,
                example,
                result,
            )

            success += 1

        except (
            torch.cuda.OutOfMemoryError
        ) as e:

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            failures.append(
                {
                    "example_id":
                        example[
                            "example_id"
                        ],

                    "error":
                        "CUDA OOM: "
                        + str(e),
                }
            )

            log()
            log(
                "ERROR on "
                f"{example['example_id']}: "
                "CUDA OOM"
            )

        except Exception as e:

            failures.append(
                {
                    "example_id":
                        example[
                            "example_id"
                        ],

                    "error":
                        str(e),
                }
            )

            log()
            log(
                "ERROR on "
                f"{example['example_id']}: "
                f"{e}"
            )

    # ========================================================
    # Save extraction report
    # ========================================================

    report_dir = (
        Path(output_root)
        / model_name
        / dataset_name
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "model":
            model_name,

        "dataset":
            dataset_name,

        "requested":
            len(examples),

        "success":
            success,

        "skipped":
            skipped,

        "failed":
            len(failures),

        "failures":
            failures,
    }

    with open(
        report_dir
        / "extraction_report.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    log()
    log(
        "Finished "
        f"{model_name} X "
        f"{dataset_name}"
    )

    log(
        f"  success={success}"
    )

    log(
        f"  skipped={skipped}"
    )

    log(
        f"  failed={len(failures)}"
    )

    return len(
        failures
    )


# ============================================================
# MODEL CLEANUP
# ============================================================

def unload_model(bundle):
    """
    Free one model before loading the next model.
    """

    if bundle is not None:

        try:
            model = bundle.get(
                "model",
                None,
            )

            if model is not None:
                del model

        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# CLI HELPERS
# ============================================================

def parse_selection(
    requested,
    available,
):
    """
    Resolve selections such as:

        --models all

    or:

        --models openbiollm medgemma
    """

    requested = list(
        requested
    )

    available = list(
        available
    )

    if requested == [
        "all"
    ]:
        return available

    unknown = (
        set(requested)
        -
        set(available)
    )

    if unknown:
        raise ValueError(
            "Unknown selection(s): "
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

    parser = (
        argparse.ArgumentParser(
            description=(
                "Extract last-prompt-token hidden "
                "activations for BRIDGE."
            )
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "all"
        ],
        help=(
            "Models to process. "
            "Use 'all' or list model names."
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "all"
        ],
        help=(
            "Datasets to process. "
            "Options: healthbench, clindet, all."
        ),
    )

    parser.add_argument(
        "--output-root",
        default=(
            DEFAULT_OUTPUT_ROOT
        ),
    )

    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=None,
        help=(
            "Smoke-test mode. "
            "Example: 1"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing activation files."
        ),
    )

    args = (
        parser.parse_args()
    )

    selected_models = (
        parse_selection(
            args.models,
            MODEL_SPECS.keys(),
        )
    )

    selected_datasets = (
        parse_selection(
            args.datasets,
            [
                "healthbench",
                "clindet",
            ],
        )
    )

    log(
        "Selected models:"
    )

    for model_name in (
        selected_models
    ):
        log(
            f"  - {model_name}"
        )

    log(
        "Selected datasets:"
    )

    for dataset_name in (
        selected_datasets
    ):
        log(
            f"  - {dataset_name}"
        )

    # Global failure counter.
    total_failures = 0

    # Load only one model at a time.
    for model_name in (
        selected_models
    ):

        bundle = None

        try:

            bundle = (
                load_model_bundle(
                    model_name
                )
            )

            for dataset_name in (
                selected_datasets
            ):

                failures = (
                    process_dataset(
                        bundle=bundle,

                        dataset_name=
                            dataset_name,

                        output_root=
                            args.output_root,

                        max_examples_per_split=
                            args.max_examples_per_split,

                        overwrite=
                            args.overwrite,
                    )
                )

                total_failures += (
                    failures
                )

        finally:

            unload_model(
                bundle
            )

            bundle = None

    # Any failed example should fail the whole process.
    if total_failures > 0:

        log()
        log("=" * 70)

        log(
            "EXTRACTION FINISHED WITH "
            f"{total_failures} FAILED EXAMPLE(S)"
        )

        log("=" * 70)

        raise SystemExit(
            1
        )

    log()
    log("=" * 70)

    log(
        "ALL REQUESTED EXTRACTIONS COMPLETE"
    )

    log("=" * 70)


if __name__ == "__main__":
    main()