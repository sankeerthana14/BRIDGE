#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Re-grade cached BRIDGE generations with a new HealthBench LLM judge.

NO target-model generation is performed.

By default, for each target medical model this grades:

1. Final unified mitigation:
   outputs/mitigation_unified_llm_judge_v1/<model>/test/
       mitigation_generations.jsonl

2. Dedicated M2-only experiment:
   outputs/mitigation_m2_v4_canonical/<model>/test/
       m2_generations.jsonl

The script:
- uses the exact OpenAI simple-evals HealthBench GRADER_TEMPLATE;
- uses HealthBench calculate_score;
- supports ordinary causal LMs and Gemma 3 judges;
- grades each unique prompt-response pair only once;
- recomputes:
    response_scores.csv
    summary.csv
    paired_vs_baseline.csv
    all_pairwise_deltas.csv

IMPORTANT:
Choose the final judge using independent physician meta-evaluation performance,
NOT based on which judge gives BRIDGE the most favorable results.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import itertools
import json
import re
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


MODELS = [
    "biomistral",
    "openbiollm",
    "ultramedical",
    "medgemma",
    "lingshu",
]

REPORT_METRICS = [
    "overall_score",
    "axis:context_awareness",
    "axis:accuracy",
    "axis:completeness",
    "axis:instruction_following",
    "axis:communication_quality",
]


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--project-root",
        type=Path,
        default=Path.home() / "JBHI" / "BRIDGE",
    )

    p.add_argument(
        "--target-model",
        required=True,
        choices=MODELS,
    )

    p.add_argument(
        "--judge-model",
        required=True,
        help="HF model ID or local model path",
    )

    p.add_argument(
        "--judge-backend",
        choices=["auto", "causal_lm", "gemma3"],
        default="auto",
    )

    p.add_argument(
        "--healthbench-jsonl",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--simple-evals-repo",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    p.add_argument(
        "--max-input-tokens",
        type=int,
        default=0,
        help="0 = no truncation",
    )

    p.add_argument(
        "--max-json-retries",
        type=int,
        default=2,
    )

    p.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
    )

    p.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--baseline-condition",
        default="no_intervention",
    )

    p.add_argument(
        "--max-unique-responses",
        type=int,
        default=0,
        help="Smoke-test cap. 0 = all.",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    p.add_argument(
        "--force",
        action="store_true",
    )

    p.add_argument(
        "--trust-remote-code",
        action="store_true",
    )

    return p.parse_args()


# =============================================================================
# Generic utilities
# =============================================================================

def norm(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()


def sha256_text(x):
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def slugify(x):
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9._-]+", "_", x)
    return x.strip("_")


def read_jsonl(path):
    rows = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception as exc:
                raise ValueError(
                    f"Bad JSON at {path}:{line_no}: {exc}"
                ) from exc

            rows.append(row)

    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


# =============================================================================
# Canonical prompt handling
# =============================================================================

def flatten_content(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        out = []

        for item in content:
            if isinstance(item, str):
                out.append(item)

            elif isinstance(item, dict):
                if item.get("type") == "text":
                    out.append(str(item.get("text", "")))

                elif item.get("content") is not None:
                    out.append(str(item["content"]))

                elif item.get("text") is not None:
                    out.append(str(item["text"]))

        return "\n".join(x for x in out if x)

    if isinstance(content, dict):
        if content.get("text") is not None:
            return str(content["text"])

        if content.get("content") is not None:
            return str(content["content"])

    return str(content or "")


def maybe_parse_structured_string(value):
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


def prompt_to_messages(value):
    if isinstance(value, str):
        parsed = maybe_parse_structured_string(value)

        if parsed is not value and not isinstance(parsed, str):
            return prompt_to_messages(parsed)

        return [
            {
                "role": "user",
                "content": value,
            }
        ]

    if isinstance(value, dict):
        if "role" in value and "content" in value:
            return [
                {
                    "role": str(
                        value.get("role") or "user"
                    ).strip().lower(),
                    "content": flatten_content(
                        value.get("content")
                    ),
                }
            ]

        for key in [
            "prompt",
            "messages",
            "conversation",
        ]:
            if key in value:
                return prompt_to_messages(
                    value[key]
                )

    if isinstance(value, list):
        out = []

        for item in value:
            if (
                isinstance(item, dict)
                and (
                    "role" in item
                    or "content" in item
                )
            ):
                out.append(
                    {
                        "role": str(
                            item.get("role") or "user"
                        ).strip().lower(),
                        "content": flatten_content(
                            item.get("content")
                        ),
                    }
                )

            elif isinstance(item, str):
                out.append(
                    {
                        "role": "user",
                        "content": item,
                    }
                )

            else:
                out.extend(
                    prompt_to_messages(item)
                )

        if out:
            return out

    return [
        {
            "role": "user",
            "content": str(value or ""),
        }
    ]


def canonicalize_messages(messages):
    allowed = {
        "system",
        "user",
        "assistant",
    }

    out = []

    for m in messages:
        role = str(
            m.get("role") or "user"
        ).strip().lower()

        if role not in allowed:
            role = "user"

        out.append(
            {
                "role": role,
                "content": str(
                    m.get("content") or ""
                ),
            }
        )

    return out


def messages_fingerprint(messages):
    clean = []

    for m in canonicalize_messages(messages):
        clean.append(
            {
                "role": m["role"],
                "content": norm(
                    flatten_content(
                        m["content"]
                    )
                ),
            }
        )

    return sha256_text(
        json.dumps(
            clean,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


# =============================================================================
# Load OpenAI simple-evals HealthBench implementation
# =============================================================================

def locate_healthbench_eval(repo):
    repo = Path(repo)

    candidates = [
        repo / "healthbench_eval.py",
        repo
        / "simple_evals"
        / "healthbench_eval.py",
    ]

    for p in candidates:
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        f"healthbench_eval.py not found under {repo}"
    )


def load_healthbench_reference(repo):
    hb_file = locate_healthbench_eval(
        repo
    )

    package_name = "bridge_simple_evals_ref"

    if package_name not in sys.modules:
        pkg = types.ModuleType(
            package_name
        )

        pkg.__path__ = [
            str(hb_file.parent)
        ]

        pkg.__package__ = (
            package_name
        )

        sys.modules[
            package_name
        ] = pkg

    module_name = (
        f"{package_name}.healthbench_eval"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        hb_file,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {hb_file}"
        )

    hb = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        module_name
    ] = hb

    spec.loader.exec_module(
        hb
    )

    return hb, hb_file


# =============================================================================
# HealthBench source + manifest mapping
# =============================================================================

def load_healthbench_examples(
    hb,
    path,
):
    rows = []
    id_map = {}
    fingerprint_map = {}

    for source_index, raw in enumerate(
        read_jsonl(path)
    ):
        ex = dict(raw)

        if (
            "prompt" not in ex
            or "rubrics" not in ex
        ):
            raise KeyError(
                f"HealthBench row {source_index} "
                "missing prompt/rubrics"
            )

        ex["rubrics"] = [
            hb.RubricItem.from_dict(r)
            for r in ex["rubrics"]
        ]

        rows.append(ex)

        for field in [
            "prompt_id",
            "example_id",
            "id",
        ]:
            if ex.get(field) is not None:
                id_map[
                    str(ex[field])
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

    return {
        "rows": rows,
        "id_map": id_map,
        "fingerprint_map": fingerprint_map,
    }


def load_manifest(path):
    df = pd.read_csv(path)

    lookup = {}

    for _, row in df.iterrows():
        key = str(
            row["example_id"]
        )

        lookup[
            key
        ] = row.to_dict()

    return lookup


def resolve_example(
    response_row,
    examples,
    manifest,
):
    rows = examples["rows"]

    # ------------------------------------------------------------
    # 1. Direct HealthBench ID
    # ------------------------------------------------------------

    for field in [
        "prompt_id",
        "example_id",
    ]:
        value = response_row.get(
            field
        )

        if (
            value is not None
            and str(value)
            in examples["id_map"]
        ):
            idx = examples[
                "id_map"
            ][str(value)]

            return (
                rows[idx],
                idx,
                "direct_id",
            )

    # ------------------------------------------------------------
    # 2. Immutable manifest prompt fingerprint
    # ------------------------------------------------------------

    manifest_id = str(
        response_row.get(
            "example_id",
            response_row.get(
                "prompt_id",
                "",
            ),
        )
    )

    if manifest_id not in manifest:
        raise KeyError(
            f"{manifest_id!r} not found in manifest"
        )

    mrow = manifest[
        manifest_id
    ]

    target_fp = messages_fingerprint(
        prompt_to_messages(
            mrow["prompt"]
        )
    )

    matches = examples[
        "fingerprint_map"
    ].get(
        target_fp,
        [],
    )

    if len(matches) == 1:
        idx = matches[0]

        return (
            rows[idx],
            idx,
            "prompt_fingerprint",
        )

    # ------------------------------------------------------------
    # 3. source_line fallback
    # ------------------------------------------------------------

    source_line = response_row.get(
        "source_line"
    )

    if source_line is None:
        source_line = mrow.get(
            "source_line"
        )

    if (
        source_line is not None
        and not pd.isna(source_line)
    ):
        line = int(
            source_line
        )

        for convention, idx in [
            (
                "source_line_one_based",
                line - 1,
            ),
            (
                "source_line_zero_based",
                line,
            ),
        ]:
            if 0 <= idx < len(rows):

                candidate_fp = (
                    messages_fingerprint(
                        prompt_to_messages(
                            rows[idx]["prompt"]
                        )
                    )
                )

                if candidate_fp == target_fp:
                    return (
                        rows[idx],
                        idx,
                        convention,
                    )

    raise KeyError(
        "Could not map generated response "
        "to HealthBench rubric row. "
        f"ID={manifest_id!r}, "
        f"source_line={source_line!r}"
    )


# =============================================================================
# Grader prompt + HealthBench score
# =============================================================================

def build_grader_prompt(
    hb,
    prompt_messages,
    response_text,
    rubric_item,
):
    convo = prompt_to_messages(
        prompt_messages
    )

    convo.append(
        {
            "role": "assistant",
            "content": response_text,
        }
    )

    convo_str = "\n\n".join(
        f"{m['role']}: {m['content']}"
        for m in convo
    )

    return (
        hb.GRADER_TEMPLATE
        .replace(
            "<<conversation>>",
            convo_str,
        )
        .replace(
            "<<rubric_item>>",
            str(rubric_item),
        )
    )


def compute_metrics(
    hb,
    example_tags,
    rubric_items,
    grades,
):
    overall = hb.calculate_score(
        rubric_items,
        grades,
    )

    if overall is None:
        raise RuntimeError(
            "HealthBench score returned None"
        )

    metrics = {
        "overall_score": float(
            overall
        )
    }

    # Example-level HealthBench tags
    for tag in example_tags or []:
        metrics[
            str(tag)
        ] = float(
            overall
        )

    # Rubric-axis tags
    tag_map = {}

    for rubric, grade in zip(
        rubric_items,
        grades,
        strict=True,
    ):
        for tag in rubric.tags:
            tag_map.setdefault(
                str(tag),
                [],
            ).append(
                (
                    rubric,
                    grade,
                )
            )

    for tag, pairs in tag_map.items():
        items = [
            x[0]
            for x in pairs
        ]

        item_grades = [
            x[1]
            for x in pairs
        ]

        score = hb.calculate_score(
            items,
            item_grades,
        )

        if score is not None:
            metrics[
                tag
            ] = float(
                score
            )

    return metrics


# =============================================================================
# Judge loader
# =============================================================================

@dataclass
class JudgeRuntime:
    model: Any
    tokenizer: Any
    renderer: Any
    model_type: str
    backend: str


def get_dtype():
    if (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16

    return torch.float16


def load_judge(
    model_name,
    backend_request,
    trust_remote_code=False,
):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    torch.cuda.set_device(0)

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    model_type = str(
        getattr(
            config,
            "model_type",
            "",
        )
    )

    if backend_request == "auto":
        backend = (
            "gemma3"
            if model_type == "gemma3"
            else "causal_lm"
        )
    else:
        backend = backend_request

    dtype = get_dtype()

    common_kwargs = {
        "dtype": dtype,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }

    print(
        "Judge architecture:",
        model_type,
    )

    print(
        "Judge backend:",
        backend,
    )

    if backend == "gemma3":

        from transformers import (
            AutoProcessor,
            Gemma3ForConditionalGeneration,
        )

        processor = (
            AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
        )

        tokenizer = (
            processor.tokenizer
        )

        model = (
            Gemma3ForConditionalGeneration
            .from_pretrained(
                model_name,
                **common_kwargs,
            )
        )

        renderer = processor

    else:

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
        )

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                model_name,
                **common_kwargs,
            )
        )

        renderer = tokenizer

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = (
        "left"
    )

    model.eval()

    return JudgeRuntime(
        model=model,
        tokenizer=tokenizer,
        renderer=renderer,
        model_type=model_type,
        backend=backend,
    )


def render_judge_prompt(
    runtime,
    prompt,
):
    if runtime.backend == "gemma3":

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        return (
            runtime.renderer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    if getattr(
        runtime.tokenizer,
        "chat_template",
        None,
    ):
        return (
            runtime.tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    return (
        "USER:\n"
        + prompt
        + "\n\nASSISTANT:\n"
    )


def model_device(model):
    try:
        return (
            model
            .get_input_embeddings()
            .weight
            .device
        )

    except Exception:
        return next(
            model.parameters()
        ).device


def generate_outputs(
    runtime,
    prompts,
    max_new_tokens,
    max_input_tokens,
):
    rendered = [
        render_judge_prompt(
            runtime,
            p,
        )
        for p in prompts
    ]

    kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "add_special_tokens": False,
    }

    if max_input_tokens > 0:
        runtime.tokenizer.truncation_side = (
            "left"
        )

        kwargs[
            "truncation"
        ] = True

        kwargs[
            "max_length"
        ] = max_input_tokens

    batch = runtime.tokenizer(
        rendered,
        **kwargs,
    )

    device = model_device(
        runtime.model
    )

    batch = {
        k: v.to(device)
        for k, v in batch.items()
    }

    input_width = (
        batch[
            "input_ids"
        ].shape[1]
    )

    generation_kwargs = {
        **batch,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": (
            runtime.tokenizer.pad_token_id
        ),
    }

    with torch.inference_mode():
        output = runtime.model.generate(
            **generation_kwargs
        )

    decoded = []

    for row in output:
        decoded.append(
            runtime.tokenizer.decode(
                row[
                    input_width:
                ],
                skip_special_tokens=True,
            ).strip()
        )

    return decoded


def generate_oom_safe(
    runtime,
    prompts,
    max_new_tokens,
    max_input_tokens,
):
    try:
        return generate_outputs(
            runtime,
            prompts,
            max_new_tokens,
            max_input_tokens,
        )

    except torch.OutOfMemoryError:

        if len(prompts) == 1:
            raise

        torch.cuda.empty_cache()

        midpoint = (
            len(prompts)
            // 2
        )

        return (
            generate_oom_safe(
                runtime,
                prompts[
                    :midpoint
                ],
                max_new_tokens,
                max_input_tokens,
            )
            +
            generate_oom_safe(
                runtime,
                prompts[
                    midpoint:
                ],
                max_new_tokens,
                max_input_tokens,
            )
        )


def parse_judge_output(
    hb,
    text,
):
    # First try official parser
    try:
        parsed = (
            hb.parse_json_to_dict(
                text
            )
        )

        if (
            isinstance(parsed, dict)
            and parsed.get(
                "criteria_met"
            )
            in (
                True,
                False,
            )
        ):
            parsed[
                "raw_judge_output"
            ] = text

            return parsed

    except Exception:
        pass

    # JSON fallback
    cleaned = str(
        text
    ).strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.I,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    first = cleaned.find(
        "{"
    )

    last = cleaned.rfind(
        "}"
    )

    candidates = [
        cleaned
    ]

    if (
        first >= 0
        and last > first
    ):
        candidates.append(
            cleaned[
                first:last + 1
            ]
        )

    for candidate in candidates:

        try:
            obj = json.loads(
                candidate
            )

            if obj.get(
                "criteria_met"
            ) in (
                True,
                False,
            ):
                obj[
                    "raw_judge_output"
                ] = text

                return obj

        except Exception:
            pass

    # Regex fallback
    match = re.search(
        r'["\']?criteria_met["\']?\s*:\s*(true|false)',
        cleaned,
        flags=re.I,
    )

    if match:
        return {
            "criteria_met": (
                match.group(1)
                .lower()
                == "true"
            ),
            "raw_judge_output": text,
        }

    return {}


class HealthBenchJudge:

    def __init__(
        self,
        model_name,
        backend,
        batch_size,
        max_new_tokens,
        max_input_tokens,
        max_retries,
        trust_remote_code=False,
    ):
        self.runtime = load_judge(
            model_name,
            backend,
            trust_remote_code,
        )

        self.batch_size = (
            batch_size
        )

        self.max_new_tokens = (
            max_new_tokens
        )

        self.max_input_tokens = (
            max_input_tokens
        )

        self.max_retries = (
            max_retries
        )

    def grade(
        self,
        prompts,
        hb,
    ):
        results = [
            None
        ] * len(prompts)

        pending = list(
            range(
                len(prompts)
            )
        )

        attempt = 0

        while pending:

            current = pending
            pending = []

            for start in range(
                0,
                len(current),
                self.batch_size,
            ):

                indices = current[
                    start:
                    start + self.batch_size
                ]

                batch_prompts = [
                    prompts[i]
                    for i in indices
                ]

                if attempt > 0:
                    batch_prompts = [
                        p
                        + "\n\nReturn exactly one valid JSON object "
                        + 'with "criteria_met" as true or false.'
                        for p in batch_prompts
                    ]

                outputs = (
                    generate_oom_safe(
                        self.runtime,
                        batch_prompts,
                        self.max_new_tokens,
                        self.max_input_tokens,
                    )
                )

                for idx, raw in zip(
                    indices,
                    outputs,
                    strict=True,
                ):

                    parsed = (
                        parse_judge_output(
                            hb,
                            raw,
                        )
                    )

                    if parsed.get(
                        "criteria_met"
                    ) in (
                        True,
                        False,
                    ):

                        parsed.setdefault(
                            "explanation",
                            "No explanation provided",
                        )

                        results[
                            idx
                        ] = parsed

                    else:
                        pending.append(
                            idx
                        )

            attempt += 1

            if (
                pending
                and attempt
                > self.max_retries
            ):
                raise RuntimeError(
                    "Judge failed to return "
                    "valid criteria_met JSON. "
                    f"Failed indices: {pending[:10]}"
                )

        return results

    def close(self):
        del self.runtime.model

        torch.cuda.empty_cache()


# =============================================================================
# Discover cached generation experiments
# =============================================================================

def discover_generation_files(
    project_root,
    model,
):
    candidates = {
        "unified_mitigation": (
            project_root
            / "outputs"
            / "mitigation_unified_llm_judge_v1"
            / model
            / "test"
            / "mitigation_generations.jsonl"
        ),

        "m2_only_v4": (
            project_root
            / "outputs"
            / "mitigation_m2_v4_canonical"
            / model
            / "test"
            / "m2_generations.jsonl"
        ),
    }

    found = {}

    for name, path in candidates.items():

        if path.exists():
            found[
                name
            ] = path

    return found


# =============================================================================
# Flatten all experiment rows + deduplicate judge work
# =============================================================================

def prepare_responses(
    sources,
    examples,
    manifest,
    model,
):
    condition_rows = []
    unique_tasks = {}

    for source_name, path in sources.items():

        rows = read_jsonl(
            path
        )

        print(
            f"{source_name}: "
            f"{len(rows)} rows"
        )

        for row_index, row in enumerate(
            rows
        ):

            if str(
                row["model"]
            ) != model:
                raise ValueError(
                    f"Wrong model in {path}"
                )

            ex, hb_index, resolution = (
                resolve_example(
                    row,
                    examples,
                    manifest,
                )
            )

            response_text = str(
                row.get(
                    "response_text"
                )
                or ""
            )

            response_hash = sha256_text(
                response_text
            )

            label = row.get(
                "evidence_label"
            )

            if label is None:
                manifest_id = str(
                    row.get(
                        "example_id",
                        row.get(
                            "prompt_id"
                        ),
                    )
                )

                label = manifest[
                    manifest_id
                ][
                    "binary_label"
                ]

            unique_key = (
                hb_index,
                response_hash,
            )

            if unique_key not in unique_tasks:

                unique_tasks[
                    unique_key
                ] = {
                    "hb_index": hb_index,
                    "response_sha256": response_hash,
                    "response_text": response_text,
                    "source_name": source_name,
                    "condition": row[
                        "condition"
                    ],
                }

            condition_rows.append(
                {
                    "source_group": source_name,
                    "hb_index": hb_index,
                    "resolution": resolution,
                    "model": model,
                    "condition": row[
                        "condition"
                    ],
                    "evidence_label": int(
                        label
                    ),
                    "response_sha256": (
                        response_hash
                    ),
                    "response_text": (
                        response_text
                    ),
                }
            )

    return (
        pd.DataFrame(
            condition_rows
        ),
        unique_tasks,
    )


# =============================================================================
# Cache unique response grades
# =============================================================================

def load_grade_cache(
    path,
    judge_model,
):
    cache = {}

    if not path.exists():
        return cache

    for row in read_jsonl(
        path
    ):

        if str(
            row.get(
                "judge_model"
            )
        ) != str(
            judge_model
        ):
            continue

        key = (
            int(
                row[
                    "hb_index"
                ]
            ),
            row[
                "response_sha256"
            ],
        )

        cache[
            key
        ] = row

    return cache


def grade_unique_responses(
    args,
    hb,
    examples,
    unique_tasks,
    grade_cache_path,
):
    cache = load_grade_cache(
        grade_cache_path,
        args.judge_model,
    )

    pending = []

    for key, task in unique_tasks.items():

        if key not in cache:
            pending.append(
                task
            )

    pending.sort(
        key=lambda x: (
            x["hb_index"],
            x["response_sha256"],
        )
    )

    if (
        args.max_unique_responses > 0
    ):
        pending = pending[
            :args.max_unique_responses
        ]

    print()
    print(
        "Unique response pairs:",
        len(unique_tasks),
    )

    print(
        "Cached:",
        len(cache),
    )

    print(
        "Pending:",
        len(pending),
    )

    if not pending:
        return cache

    judge = HealthBenchJudge(
        model_name=args.judge_model,
        backend=args.judge_backend,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        max_retries=args.max_json_retries,
        trust_remote_code=args.trust_remote_code,
    )

    try:

        for i, task in enumerate(
            pending,
            start=1,
        ):

            hb_index = task[
                "hb_index"
            ]

            example = examples[
                "rows"
            ][
                hb_index
            ]

            rubrics = example[
                "rubrics"
            ]

            prompts = [
                build_grader_prompt(
                    hb,
                    example[
                        "prompt"
                    ],
                    task[
                        "response_text"
                    ],
                    rubric,
                )
                for rubric in rubrics
            ]

            print(
                f"[{i}/{len(pending)}] "
                f"hb={hb_index} | "
                f"{task['source_name']} | "
                f"{task['condition']} | "
                f"{len(rubrics)} rubrics",
                flush=True,
            )

            start = time.time()

            grades = judge.grade(
                prompts,
                hb,
            )

            metrics = compute_metrics(
                hb,
                example.get(
                    "example_tags",
                    [],
                ),
                rubrics,
                grades,
            )

            result = {
                "hb_index": hb_index,
                "response_sha256": (
                    task[
                        "response_sha256"
                    ]
                ),
                "response_text": (
                    task[
                        "response_text"
                    ]
                ),
                "judge_model": (
                    args.judge_model
                ),
                "judge_backend": (
                    judge.runtime.backend
                ),
                "grading_seconds": (
                    time.time()
                    - start
                ),
                "metrics": metrics,
                "rubric_grades": grades,
            }

            append_jsonl(
                grade_cache_path,
                result,
            )

            cache[
                (
                    hb_index,
                    task[
                        "response_sha256"
                    ],
                )
            ] = result

    finally:
        judge.close()

    return cache


# =============================================================================
# Expand unique grades to condition-level table
# =============================================================================

def expand_scores(
    condition_df,
    cache,
):
    rows = []

    for _, row in condition_df.iterrows():

        key = (
            int(
                row[
                    "hb_index"
                ]
            ),
            row[
                "response_sha256"
            ],
        )

        grade = cache.get(
            key
        )

        # During smoke test many rows will intentionally be missing
        if grade is None:
            continue

        flat = row.to_dict()

        for metric, value in (
            grade[
                "metrics"
            ].items()
        ):

            flat[
                metric
            ] = value

        rows.append(
            flat
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Bootstrap statistics
# =============================================================================

def clipped_mean(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return np.nan

    return float(
        np.clip(
            np.mean(x),
            0.0,
            1.0,
        )
    )


def bootstrap_mean(
    values,
    n_boot,
    rng,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    point = clipped_mean(
        values
    )

    boots = []

    for _ in range(
        n_boot
    ):

        idx = rng.integers(
            0,
            len(values),
            len(values),
        )

        boots.append(
            clipped_mean(
                values[idx]
            )
        )

    low, high = np.percentile(
        boots,
        [
            2.5,
            97.5,
        ],
    )

    return (
        point,
        low,
        high,
    )


def bootstrap_delta(
    candidate,
    baseline,
    n_boot,
    rng,
):
    a = np.asarray(
        candidate,
        dtype=float,
    )

    b = np.asarray(
        baseline,
        dtype=float,
    )

    mask = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    a = a[
        mask
    ]

    b = b[
        mask
    ]

    point = (
        clipped_mean(a)
        - clipped_mean(b)
    )

    boots = []

    for _ in range(
        n_boot
    ):

        idx = rng.integers(
            0,
            len(a),
            len(a),
        )

        boots.append(
            clipped_mean(
                a[idx]
            )
            - clipped_mean(
                b[idx]
            )
        )

    low, high = np.percentile(
        boots,
        [
            2.5,
            97.5,
        ],
    )

    return (
        point,
        low,
        high,
    )


def make_summary(
    scores,
    n_boot,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    rows = []

    for source_group, df in scores.groupby(
        "source_group"
    ):

        groups = [
            (
                "all",
                df,
            ),
            (
                "insufficient",
                df[
                    df[
                        "evidence_label"
                    ]
                    == 1
                ],
            ),
            (
                "sufficient",
                df[
                    df[
                        "evidence_label"
                    ]
                    == 0
                ],
            ),
        ]

        for label_group, subset in groups:

            for condition, g in subset.groupby(
                "condition"
            ):

                for metric in REPORT_METRICS:

                    if metric not in g:
                        continue

                    vals = pd.to_numeric(
                        g[
                            metric
                        ],
                        errors="coerce",
                    ).dropna()

                    if len(vals) == 0:
                        continue

                    mean, lo, hi = (
                        bootstrap_mean(
                            vals,
                            n_boot,
                            rng,
                        )
                    )

                    rows.append(
                        {
                            "source_group": source_group,
                            "model": g[
                                "model"
                            ].iloc[0],
                            "condition": condition,
                            "label_group": label_group,
                            "metric": metric,
                            "n": len(vals),
                            "mean": mean,
                            "ci95_low": lo,
                            "ci95_high": hi,
                        }
                    )

    return pd.DataFrame(
        rows
    )


def pair_conditions(
    df,
    candidate_condition,
    baseline_condition,
    label_group,
    metric,
):
    if label_group == "insufficient":

        df = df[
            df[
                "evidence_label"
            ]
            == 1
        ]

    elif label_group == "sufficient":

        df = df[
            df[
                "evidence_label"
            ]
            == 0
        ]

    candidate = (
        df[
            df[
                "condition"
            ]
            == candidate_condition
        ]
        .set_index(
            "hb_index"
        )
    )

    baseline = (
        df[
            df[
                "condition"
            ]
            == baseline_condition
        ]
        .set_index(
            "hb_index"
        )
    )

    common = candidate.index.intersection(
        baseline.index
    )

    if len(common) == 0:
        return pd.DataFrame()

    return pd.DataFrame(
        {
            "candidate": pd.to_numeric(
                candidate.loc[
                    common,
                    metric,
                ],
                errors="coerce",
            ),
            "baseline": pd.to_numeric(
                baseline.loc[
                    common,
                    metric,
                ],
                errors="coerce",
            ),
        }
    ).dropna()


def make_paired_vs_baseline(
    scores,
    baseline_condition,
    n_boot,
    seed,
):
    rng = np.random.default_rng(
        seed + 1
    )

    rows = []

    for source_group, df in scores.groupby(
        "source_group"
    ):

        conditions = sorted(
            df[
                "condition"
            ].unique()
        )

        if baseline_condition not in conditions:
            continue

        for condition in conditions:

            if condition == baseline_condition:
                continue

            for label_group in [
                "all",
                "insufficient",
                "sufficient",
            ]:

                for metric in REPORT_METRICS:

                    if metric not in df:
                        continue

                    pair = pair_conditions(
                        df,
                        condition,
                        baseline_condition,
                        label_group,
                        metric,
                    )

                    if len(pair) == 0:
                        continue

                    delta, lo, hi = (
                        bootstrap_delta(
                            pair[
                                "candidate"
                            ],
                            pair[
                                "baseline"
                            ],
                            n_boot,
                            rng,
                        )
                    )

                    rows.append(
                        {
                            "source_group": source_group,
                            "model": df[
                                "model"
                            ].iloc[0],
                            "condition": condition,
                            "baseline_condition": (
                                baseline_condition
                            ),
                            "label_group": label_group,
                            "metric": metric,
                            "n_paired": len(pair),
                            "delta_candidate_minus_baseline": (
                                delta
                            ),
                            "delta_ci95_low": lo,
                            "delta_ci95_high": hi,
                        }
                    )

    return pd.DataFrame(
        rows
    )


def make_all_pairwise(
    scores,
    n_boot,
    seed,
):
    rng = np.random.default_rng(
        seed + 2
    )

    rows = []

    for source_group, df in scores.groupby(
        "source_group"
    ):

        conditions = sorted(
            df[
                "condition"
            ].unique()
        )

        for a, b in itertools.combinations(
            conditions,
            2,
        ):

            for label_group in [
                "all",
                "insufficient",
                "sufficient",
            ]:

                for metric in REPORT_METRICS:

                    if metric not in df:
                        continue

                    pair = pair_conditions(
                        df,
                        a,
                        b,
                        label_group,
                        metric,
                    )

                    if len(pair) == 0:
                        continue

                    delta, lo, hi = (
                        bootstrap_delta(
                            pair[
                                "candidate"
                            ],
                            pair[
                                "baseline"
                            ],
                            n_boot,
                            rng,
                        )
                    )

                    rows.append(
                        {
                            "source_group": source_group,
                            "model": df[
                                "model"
                            ].iloc[0],
                            "condition_a": a,
                            "condition_b": b,
                            "label_group": label_group,
                            "metric": metric,
                            "n_paired": len(pair),
                            "delta_a_minus_b": delta,
                            "delta_ci95_low": lo,
                            "delta_ci95_high": hi,
                        }
                    )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()

    root = (
        args.project_root
        .expanduser()
        .resolve()
    )

    if args.healthbench_jsonl is None:

        args.healthbench_jsonl = (
            root
            / "data"
            / "healthbench"
            / "hard_2025-05-08-21-00-10.jsonl"
        )

    if args.manifest is None:

        args.manifest = (
            root
            / "data"
            / "manifests"
            / "healthbench_evidence_sufficiency_v1.csv"
        )

    if args.simple_evals_repo is None:

        args.simple_evals_repo = (
            root
            / "third_party"
            / "simple-evals"
        )

    sources = discover_generation_files(
        root,
        args.target_model,
    )

    if not sources:

        raise FileNotFoundError(
            "No cached generation experiments found."
        )

    print(
        "\nDiscovered:"
    )

    for name, path in sources.items():

        print(
            f"  {name}: {path}"
        )

    hb, hb_file = (
        load_healthbench_reference(
            args.simple_evals_repo
        )
    )

    examples = load_healthbench_examples(
        hb,
        args.healthbench_jsonl,
    )

    manifest = load_manifest(
        args.manifest
    )

    condition_df, unique_tasks = (
        prepare_responses(
            sources,
            examples,
            manifest,
            args.target_model,
        )
    )

    judge_slug = slugify(
        args.judge_model
    )

    output_dir = (
        root
        / "outputs"
        / "healthbench_regrade"
        / judge_slug
        / args.target_model
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    condition_df.to_csv(
        output_dir
        / "canonical_response_index.csv",
        index=False,
    )

    print(
        "\nCondition rows:",
        len(condition_df),
    )

    print(
        "Unique response pairs:",
        len(unique_tasks),
    )

    print(
        "Judge calls saved by deduplication:",
        len(condition_df)
        - len(unique_tasks),
    )

    if args.dry_run:

        print(
            "\nDRY RUN PASS."
        )

        return

    grade_cache_path = (
        output_dir
        / "unique_response_grades.jsonl"
    )

    if (
        args.force
        and grade_cache_path.exists()
    ):
        grade_cache_path.unlink()

    cache = grade_unique_responses(
        args,
        hb,
        examples,
        unique_tasks,
        grade_cache_path,
    )

    scores = expand_scores(
        condition_df,
        cache,
    )

    scores.to_csv(
        output_dir
        / "all_response_scores.csv",
        index=False,
    )

    summary = make_summary(
        scores,
        args.bootstrap,
        args.bootstrap_seed,
    )

    paired = make_paired_vs_baseline(
        scores,
        args.baseline_condition,
        args.bootstrap,
        args.bootstrap_seed,
    )

    all_pairwise = make_all_pairwise(
        scores,
        args.bootstrap,
        args.bootstrap_seed,
    )

    summary.to_csv(
        output_dir
        / "all_summary.csv",
        index=False,
    )

    paired.to_csv(
        output_dir
        / "all_paired_vs_baseline.csv",
        index=False,
    )

    all_pairwise.to_csv(
        output_dir
        / "all_pairwise_deltas.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Write clean separate outputs for paper experiments
    # ------------------------------------------------------------

    for source_group in sorted(
        scores[
            "source_group"
        ].unique()
    ):

        source_dir = (
            output_dir
            / "by_source"
            / source_group
        )

        source_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_scores = scores[
            scores[
                "source_group"
            ]
            == source_group
        ]

        source_scores.to_csv(
            source_dir
            / "response_scores.csv",
            index=False,
        )

        summary[
            summary[
                "source_group"
            ]
            == source_group
        ].to_csv(
            source_dir
            / "summary.csv",
            index=False,
        )

        paired[
            paired[
                "source_group"
            ]
            == source_group
        ].to_csv(
            source_dir
            / "paired_vs_baseline.csv",
            index=False,
        )

        all_pairwise[
            all_pairwise[
                "source_group"
            ]
            == source_group
        ].to_csv(
            source_dir
            / "all_pairwise_deltas.csv",
            index=False,
        )

    metadata = {
        "target_model": args.target_model,
        "judge_model": args.judge_model,
        "judge_backend": args.judge_backend,
        "healthbench_jsonl": str(
            args.healthbench_jsonl
        ),
        "manifest": str(
            args.manifest
        ),
        "healthbench_eval_py": str(
            hb_file
        ),
        "grader_template_sha256": (
            sha256_text(
                hb.GRADER_TEMPLATE
            )
        ),
        "bootstrap": args.bootstrap,
        "bootstrap_seed": (
            args.bootstrap_seed
        ),
        "n_condition_rows": (
            len(condition_df)
        ),
        "n_unique_response_pairs": (
            len(unique_tasks)
        ),
        "regex_behavior_classifier_used": False,
        "ask_abstain_direct_used": False,
    }

    (
        output_dir
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 80
    )

    print(
        "RE-GRADING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "Target model:",
        args.target_model,
    )

    print(
        "Judge:",
        args.judge_model,
    )

    print(
        "Output:",
        output_dir,
    )

    print()
    print(
        "Unified mitigation results:"
    )

    print(
        output_dir
        / "by_source"
        / "unified_mitigation"
        / "paired_vs_baseline.csv"
    )

    print()
    print(
        "M2-only results:"
    )

    print(
        output_dir
        / "by_source"
        / "m2_only_v4"
        / "paired_vs_baseline.csv"
    )


if __name__ == "__main__":
    main()