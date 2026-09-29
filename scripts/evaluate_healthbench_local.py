#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import json
import re
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

INSUFFICIENT_TAGS = {
    "physician_agreed_category:not-enough-context",
    "physician_agreed_category:not-enough-info-to-complete-task",
}
SUFFICIENT_TAGS = {
    "physician_agreed_category:enough-context",
    "physician_agreed_category:enough-info-to-complete-task",
}

REPORT_METRICS = [
    "overall_score",
    "axis:context_awareness",
    "axis:accuracy",
    "axis:completeness",
    "axis:instruction_following",
    "axis:communication_quality",
]

CAUSAL_FIELDS = [
    "evidence_label", "evidence_score", "flagged", "seed", "generation_id",
    "experiment", "layer_0based", "layer_1based", "alpha", "lambda",
    "direction_kind", "projection_scale", "probe_val_auroc",
    "intervention_scope", "manifest_id", "source_file", "source_line",
    "raw_index", "activation_norm", "cav_projection_before",
    "cav_projection_after", "intervention_projection_before",
    "intervention_projection_after", "centered_projection_before",
    "centered_projection_after", "delta_norm",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--healthbench-jsonl", required=True, type=Path)
    p.add_argument("--responses-jsonl", required=True, type=Path)
    p.add_argument("--simple-evals-repo", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--baseline-condition", default="no_intervention")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--limit-responses", type=int, default=None)
    p.add_argument("--max-json-retries", type=int, default=2)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_healthbench_reference(repo: Path):
    repo = repo.resolve()
    hb_file = repo / "healthbench_eval.py"
    if not hb_file.exists():
        raise FileNotFoundError(f"Missing {hb_file}")

    package_name = "simple_evals_ref"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(repo)]
        pkg.__package__ = package_name
        pkg.__spec__ = importlib.machinery.ModuleSpec(
            package_name, loader=None, is_package=True
        )
        sys.modules[package_name] = pkg

    return importlib.import_module(f"{package_name}.healthbench_eval")


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at {path}:{i}: {e}") from e
    return rows


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_text(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def infer_evidence_label(tags):
    tags = set(tags or [])
    ins = bool(tags & INSUFFICIENT_TAGS)
    suff = bool(tags & SUFFICIENT_TAGS)
    if ins and suff:
        raise ValueError("Example has both sufficient and insufficient target tags")
    if ins:
        return 1
    if suff:
        return 0
    return None


class LocalJudge:
    def __init__(self, model_name, batch_size, max_new_tokens, max_retries):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the local rubric judge")

        torch.cuda.set_device(0)
        self.device = torch.device("cuda:0")
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.model_name = model_name

        print(f"Loading local judge: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        print("Judge class:", self.model.__class__.__name__)
        print("Judge device:", self.device)

    def format_prompt(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return "USER:\n" + prompt + "\n\nASSISTANT:\n"

    @staticmethod
    def fallback_parse(text: str):
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        m = re.search(r"\{.*\}", cleaned, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        lowered = cleaned.lower()
        if re.search(r'"?criteria_met"?\s*:\s*true', lowered):
            return {"criteria_met": True, "explanation": cleaned}
        if re.search(r'"?criteria_met"?\s*:\s*false', lowered):
            return {"criteria_met": False, "explanation": cleaned}
        return {}

    def generate_batch(self, prompts):
        formatted = [self.format_prompt(p) for p in prompts]
        encoded = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        prompt_width = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            out = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return [
            self.tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip()
            for row in out
        ]

    def grade(self, prompts, reference_parser):
        results = [None] * len(prompts)
        pending = list(range(len(prompts)))
        attempt = 0

        while pending:
            current = pending
            pending = []

            for start in range(0, len(current), self.batch_size):
                indices = current[start:start + self.batch_size]
                batch_prompts = [prompts[i] for i in indices]
                if attempt > 0:
                    batch_prompts = [
                        p + "\n\nReturn exactly one valid JSON object with criteria_met as true or false."
                        for p in batch_prompts
                    ]

                outputs = self.generate_batch(batch_prompts)
                for idx, raw in zip(indices, outputs, strict=True):
                    parsed = reference_parser(raw)
                    if not isinstance(parsed, dict):
                        parsed = {}
                    if parsed.get("criteria_met") not in (True, False):
                        parsed = self.fallback_parse(raw)

                    if parsed.get("criteria_met") in (True, False):
                        parsed.setdefault("explanation", "No explanation provided")
                        parsed["raw_judge_output"] = raw
                        results[idx] = parsed
                    else:
                        pending.append(idx)

            attempt += 1
            if pending and attempt > self.max_retries:
                raise RuntimeError(
                    f"Judge failed to return valid criteria_met JSON after {attempt} attempts. "
                    f"First failed rubric indices: {pending[:10]}"
                )

        return results


def build_grader_prompt(hb, prompt_messages, response_text, rubric_item):
    convo = prompt_messages + [{"role": "assistant", "content": response_text}]
    convo_str = "\n\n".join(f"{m['role']}: {m['content']}" for m in convo)
    return hb.GRADER_TEMPLATE.replace(
        "<<conversation>>", convo_str
    ).replace(
        "<<rubric_item>>", str(rubric_item)
    )


def compute_metrics(hb, example_tags, rubric_items, grades):
    overall = hb.calculate_score(rubric_items, grades)
    if overall is None:
        raise RuntimeError("HealthBench overall score unexpectedly returned None")

    metrics = {"overall_score": overall}
    for tag in example_tags:
        metrics[tag] = overall

    tag_map = {}
    for rubric_item, grade in zip(rubric_items, grades, strict=True):
        for tag in rubric_item.tags:
            tag_map.setdefault(tag, []).append((rubric_item, grade))

    for tag, pairs in tag_map.items():
        items = [x[0] for x in pairs]
        item_grades = [x[1] for x in pairs]
        score = hb.calculate_score(items, item_grades)
        if score is not None:
            metrics[tag] = score

    return metrics


def cache_key(row, judge_model):
    return (
        str(row["prompt_id"]),
        str(row["model"]),
        str(row["condition"]),
        sha256_text(row["response_text"]),
        judge_model,
    )


def clipped_mean(values):
    return float(np.clip(np.mean(values), 0.0, 1.0))


def bootstrap_mean_ci(values, n_boot, rng):
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


def paired_bootstrap_delta(a, b, n_boot, rng):
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
        boots[i] = clipped_mean(a[idx]) - clipped_mean(b[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def make_summary(scores, out_path, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    groups = [
        ("all", scores),
        ("insufficient", scores[scores["evidence_label"] == 1]),
        ("sufficient", scores[scores["evidence_label"] == 0]),
    ]
    for label_group, subset in groups:
        for (model, condition), g in subset.groupby(["model", "condition"]):
            for metric in REPORT_METRICS:
                if metric not in g.columns:
                    continue
                vals = pd.to_numeric(g[metric], errors="coerce").to_numpy()
                vals = vals[np.isfinite(vals)]
                if not len(vals):
                    continue
                mean, lo, hi = bootstrap_mean_ci(vals, n_boot, rng)
                rows.append({
                    "model": model,
                    "condition": condition,
                    "label_group": label_group,
                    "metric": metric,
                    "n": len(vals),
                    "mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def make_paired(scores, out_path, baseline_condition, n_boot, seed):
    rng = np.random.default_rng(seed + 1)
    rows = []
    groups = [
        ("all", scores),
        ("insufficient", scores[scores["evidence_label"] == 1]),
        ("sufficient", scores[scores["evidence_label"] == 0]),
    ]
    for label_group, subset in groups:
        for model, model_df in subset.groupby("model"):
            conditions = sorted(model_df["condition"].dropna().unique())
            if baseline_condition not in conditions:
                continue
            base = model_df[model_df["condition"] == baseline_condition].set_index("prompt_id")
            for condition in conditions:
                if condition == baseline_condition:
                    continue
                cand = model_df[model_df["condition"] == condition].set_index("prompt_id")
                common = base.index.intersection(cand.index)
                if not len(common):
                    continue
                for metric in REPORT_METRICS:
                    if metric not in base.columns or metric not in cand.columns:
                        continue
                    pair = pd.DataFrame({
                        "candidate": pd.to_numeric(cand.loc[common, metric], errors="coerce"),
                        "baseline": pd.to_numeric(base.loc[common, metric], errors="coerce"),
                    }).dropna()
                    if not len(pair):
                        continue
                    delta, lo, hi = paired_bootstrap_delta(
                        pair["candidate"].to_numpy(),
                        pair["baseline"].to_numpy(),
                        n_boot,
                        rng,
                    )
                    rows.append({
                        "model": model,
                        "condition": condition,
                        "baseline_condition": baseline_condition,
                        "label_group": label_group,
                        "metric": metric,
                        "n_paired": len(pair),
                        "delta_candidate_minus_baseline": delta,
                        "delta_ci95_low": lo,
                        "delta_ci95_high": hi,
                    })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grades_path = args.output_dir / "rubric_grades.jsonl"
    scores_path = args.output_dir / "response_scores.csv"
    summary_path = args.output_dir / "summary.csv"
    paired_path = args.output_dir / "paired_vs_baseline.csv"
    metadata_path = args.output_dir / "metadata.json"

    hb = load_healthbench_reference(args.simple_evals_repo)

    examples = {}
    for ex in read_jsonl(args.healthbench_jsonl):
        ex = dict(ex)
        ex["rubrics"] = [hb.RubricItem.from_dict(r) for r in ex["rubrics"]]
        examples[str(ex["prompt_id"])] = ex

    responses = read_jsonl(args.responses_jsonl)
    required = {"prompt_id", "model", "condition", "response_text"}
    for i, row in enumerate(responses):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Response row {i} missing fields: {sorted(missing)}")
        if str(row["prompt_id"]) not in examples:
            raise KeyError(f"prompt_id {row['prompt_id']} not found in HealthBench file")

    existing = read_jsonl(grades_path)
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
        r for r in responses
        if args.force or cache_key(r, args.judge_model) not in existing_keys
    ]
    if args.limit_responses is not None:
        pending = pending[:args.limit_responses]

    print("=" * 80)
    print("LOCAL HEALTHBENCH CAUSAL EVALUATOR")
    print("=" * 80)
    print("HealthBench:", args.healthbench_jsonl)
    print("Responses:", args.responses_jsonl)
    print("Judge:", args.judge_model)
    print("Cached responses:", len(existing))
    print("Pending responses:", len(pending))
    print("=" * 80)

    judge = None
    if pending:
        judge = LocalJudge(
            args.judge_model,
            args.batch_size,
            args.max_new_tokens,
            args.max_json_retries,
        )

    for i, row in enumerate(pending, 1):
        prompt_id = str(row["prompt_id"])
        ex = examples[prompt_id]
        label = row.get("evidence_label")
        if label is None:
            label = infer_evidence_label(ex.get("example_tags", []))

        rubrics = ex["rubrics"]
        prompts = [
            build_grader_prompt(hb, ex["prompt"], row["response_text"], rubric)
            for rubric in rubrics
        ]

        print(
            f"[{i}/{len(pending)}] {row['model']} | {row['condition']} | "
            f"{prompt_id} | {len(rubrics)} rubrics"
        )
        start = time.time()
        grades = judge.grade(prompts, hb.parse_json_to_dict)
        metrics = compute_metrics(hb, ex.get("example_tags", []), rubrics, grades)

        rubric_items = []
        for rubric, grade in zip(rubrics, grades, strict=True):
            rubric_items.append({
                **rubric.to_dict(),
                "criteria_met": grade["criteria_met"],
                "explanation": grade.get("explanation", "No explanation provided"),
                "raw_judge_output": grade.get("raw_judge_output"),
            })

        result = {
            "prompt_id": prompt_id,
            "model": row["model"],
            "condition": row["condition"],
            "evidence_label": label,
            "response_text": row["response_text"],
            "response_sha256": sha256_text(row["response_text"]),
            "judge_model": args.judge_model,
            "judge_type": "local_llm",
            "grading_seconds": time.time() - start,
            "metrics": metrics,
            "rubric_items": rubric_items,
        }
        for field in CAUSAL_FIELDS:
            if field != "evidence_label":
                result[field] = row.get(field)
        append_jsonl(grades_path, result)

    all_grades = read_jsonl(grades_path)
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
        for field in CAUSAL_FIELDS:
            if field != "evidence_label":
                flat[field] = r.get(field)
        for name, value in r["metrics"].items():
            flat[name] = value
        flat_rows.append(flat)

    scores = pd.DataFrame(flat_rows)
    if not scores.empty:
        scores = scores.drop_duplicates(
            subset=["prompt_id", "model", "condition", "response_sha256", "judge_model"],
            keep="last",
        )
    scores.to_csv(scores_path, index=False)

    make_summary(scores, summary_path, args.bootstrap, args.bootstrap_seed)
    make_paired(
        scores,
        paired_path,
        args.baseline_condition,
        args.bootstrap,
        args.bootstrap_seed,
    )

    metadata = {
        "healthbench_jsonl": str(args.healthbench_jsonl),
        "responses_jsonl": str(args.responses_jsonl),
        "simple_evals_repo": str(args.simple_evals_repo),
        "judge_model": args.judge_model,
        "judge_type": "local_llm",
        "healthbench_template_source": "OpenAI simple-evals GRADER_TEMPLATE",
        "healthbench_score_source": "OpenAI simple-evals calculate_score",
        "baseline_condition": args.baseline_condition,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "bootstrap": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "num_cached_grades": len(read_jsonl(grades_path)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(" ", grades_path)
    print(" ", scores_path)
    print(" ", summary_path)
    print(" ", paired_path)
    print(" ", metadata_path)


if __name__ == "__main__":
    main()
