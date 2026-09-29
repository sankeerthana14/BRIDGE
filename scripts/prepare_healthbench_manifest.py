import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LABEL DEFINITIONS
# ============================================================

# BRIDGE definition:
# 1 = insufficient patient-specific evidence
# 0 = sufficient patient-specific evidence

INSUFFICIENT_CATEGORIES = {
    "not-enough-context",
    "not-enough-info-to-complete-task",
}

SUFFICIENT_CATEGORIES = {
    "enough-context",
    "enough-info-to-complete-task",
}

TARGET_PREFIX = "physician_agreed_category:"


# ============================================================
# 2. READ JSONL
# ============================================================

def read_jsonl(path):
    """
    Read a JSONL file.

    JSONL = one JSON object per line.

    Returns:
        [
            {
                "line_number": 1,
                "record": {...}
            },
            ...
        ]
    """

    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse JSON in {path} "
                    f"at line {line_number}: {e}"
                )

            records.append(
                {
                    "line_number": line_number,
                    "record": record,
                }
            )

    return records


# ============================================================
# 3. RECURSIVELY COLLECT STRINGS
# ============================================================

def collect_strings(obj):
    """
    Recursively walk through a nested Python object and
    collect every string found in keys and values.

    We use this to locate HealthBench category tags without
    initially assuming their exact position in the JSON schema.
    """

    strings = []

    if isinstance(obj, dict):

        for key, value in obj.items():

            if isinstance(key, str):
                strings.append(key)

            strings.extend(
                collect_strings(value)
            )

    elif isinstance(obj, list):

        for item in obj:
            strings.extend(
                collect_strings(item)
            )

    elif isinstance(obj, str):

        strings.append(obj)

    return strings


# ============================================================
# 4. EXTRACT PHYSICIAN-AGREED CATEGORIES
# ============================================================

def extract_physician_categories(record):
    """
    Find HealthBench category tags such as:

        physician_agreed_category:not-enough-context

    Returns:

        [
            "not-enough-context",
            ...
        ]
    """

    all_strings = collect_strings(record)

    categories = set()

    for text in all_strings:

        # ----------------------------------------------------
        # Case 1:
        # String itself begins with physician_agreed_category:
        # ----------------------------------------------------

        if text.startswith(TARGET_PREFIX):

            category = text[len(TARGET_PREFIX):].strip()

            if category:
                categories.add(category)

        # ----------------------------------------------------
        # Case 2:
        # Tag appears inside a longer string
        # ----------------------------------------------------

        elif TARGET_PREFIX in text:

            pieces = text.split(TARGET_PREFIX)

            for piece in pieces[1:]:

                category = (
                    piece
                    .split()[0]
                    .strip()
                    .strip(",")
                    .strip(";")
                    .strip('"')
                    .strip("'")
                    .strip("]")
                    .strip("}")
                )

                if category:
                    categories.add(category)

    return sorted(categories)


# ============================================================
# 5. EXTRACT PROMPT / CONVERSATION
# ============================================================

def extract_prompt_text(record):
    """
    Try several likely HealthBench locations for the input
    conversation.

    This text is used for:
        - overlap detection
        - later human annotation
        - error analysis

    It does NOT determine the binary label.
    """

    # --------------------------------------------------------
    # Case 1: prompt field exists
    # --------------------------------------------------------

    if "prompt" in record:

        prompt = record["prompt"]

        if isinstance(prompt, str):
            return prompt

        return json.dumps(
            prompt,
            ensure_ascii=False,
            sort_keys=True,
        )

    # --------------------------------------------------------
    # Case 2: messages-style conversation
    # --------------------------------------------------------

    if (
        "messages" in record
        and isinstance(record["messages"], list)
    ):

        parts = []

        for message in record["messages"]:

            if not isinstance(message, dict):
                continue

            role = message.get(
                "role",
                "unknown"
            )

            content = message.get(
                "content",
                ""
            )

            if isinstance(content, str):

                parts.append(
                    f"{role}: {content}"
                )

            else:

                parts.append(
                    f"{role}: "
                    + json.dumps(
                        content,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )

        return "\n".join(parts)

    # --------------------------------------------------------
    # Case 3: conversation field
    # --------------------------------------------------------

    if "conversation" in record:

        conversation = record["conversation"]

        if isinstance(conversation, str):
            return conversation

        return json.dumps(
            conversation,
            ensure_ascii=False,
            sort_keys=True,
        )

    # --------------------------------------------------------
    # If we cannot identify the prompt
    # --------------------------------------------------------

    return ""


# ============================================================
# 6. NORMALIZE PROMPT
# ============================================================

def normalize_prompt(text):
    """
    Normalize superficial formatting differences before
    checking whether OSS-Eval and HealthBench Hard contain
    the same prompt.

    We intentionally do NOT perform aggressive text cleaning.

    We only:
        - strip leading/trailing whitespace
        - collapse repeated whitespace
    """

    if not isinstance(text, str):
        return ""

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


# ============================================================
# 7. CREATE PROMPT HASH
# ============================================================

def make_prompt_hash(prompt):
    """
    Create a deterministic hash from the normalized prompt.

    This lets us compare prompts between OSS-Eval and
    HealthBench Hard without storing complex Python objects
    as identifiers.

    Returns None if no prompt could be extracted.
    """

    normalized = normalize_prompt(prompt)

    if not normalized:
        return None

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# 8. CREATE STABLE EXAMPLE ID
# ============================================================

def make_example_id(
    record,
    source_name,
    line_number,
):
    """
    Create a stable record identifier.

    Example:

        oss_eval_000123_a92fc383

    The last component is a hash of the raw JSON record.
    """

    canonical_json = json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
    )

    digest = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()[:8]

    return (
        f"{source_name}_"
        f"{line_number:06d}_"
        f"{digest}"
    )


# ============================================================
# 9. DERIVE BRIDGE BINARY LABEL
# ============================================================

def derive_label(categories):
    """
    Convert HealthBench physician-agreed categories into the
    BRIDGE evidence-sufficiency label.

    Returns:

        binary_label
        included
        exclusion_reason

    Convention:

        1 = insufficient evidence
        0 = sufficient evidence
        None = excluded
    """

    category_set = set(categories)

    insufficient_matches = (
        category_set
        & INSUFFICIENT_CATEGORIES
    )

    sufficient_matches = (
        category_set
        & SUFFICIENT_CATEGORIES
    )

    # --------------------------------------------------------
    # Clearly insufficient
    # --------------------------------------------------------

    if (
        insufficient_matches
        and not sufficient_matches
    ):

        return (
            1,
            True,
            "",
        )

    # --------------------------------------------------------
    # Clearly sufficient
    # --------------------------------------------------------

    if (
        sufficient_matches
        and not insufficient_matches
    ):

        return (
            0,
            True,
            "",
        )

    # --------------------------------------------------------
    # Ambiguous:
    # both sufficient and insufficient target labels present
    # --------------------------------------------------------

    if (
        insufficient_matches
        and sufficient_matches
    ):

        return (
            None,
            False,
            "contains both sufficient and insufficient categories",
        )

    # --------------------------------------------------------
    # Does not belong to our binary task
    # --------------------------------------------------------

    return (
        None,
        False,
        "no target evidence-sufficiency category",
    )


# ============================================================
# 10. PROCESS ONE HEALTHBENCH FILE
# ============================================================

def process_file(
    path,
    dataset_variant,
):
    """
    Create one manifest row for every raw HealthBench record.

    IMPORTANT:
    Excluded examples are kept in the manifest.
    """

    path = Path(path)

    rows = []

    records = read_jsonl(path)

    print(
        f"\nReading {dataset_variant}: "
        f"{len(records)} raw records"
    )

    missing_prompt_count = 0

    for item in records:

        line_number = item["line_number"]
        record = item["record"]

        # ----------------------------------------------------
        # Extract original HealthBench annotations
        # ----------------------------------------------------

        categories = (
            extract_physician_categories(
                record
            )
        )

        # ----------------------------------------------------
        # Convert them into BRIDGE binary task
        # ----------------------------------------------------

        (
            label,
            included,
            exclusion_reason,
        ) = derive_label(categories)

        # ----------------------------------------------------
        # Extract prompt
        # ----------------------------------------------------

        prompt = extract_prompt_text(
            record
        )

        if not prompt:
            missing_prompt_count += 1

        # ----------------------------------------------------
        # Stable identifiers
        # ----------------------------------------------------

        example_id = make_example_id(
            record=record,
            source_name=dataset_variant,
            line_number=line_number,
        )

        prompt_hash = make_prompt_hash(
            prompt
        )

        # ----------------------------------------------------
        # Store everything
        # ----------------------------------------------------

        rows.append(
            {
                "example_id": example_id,

                "source_file": path.name,

                "source_line": line_number,

                "dataset_variant":
                    dataset_variant,

                "physician_agreed_categories":
                    "|".join(categories),

                "binary_label":
                    label,

                "included":
                    included,

                "exclusion_reason":
                    exclusion_reason,

                # Set later
                "overlaps_hard_test":
                    False,

                # Set later
                "split":
                    None,

                "prompt_hash":
                    prompt_hash,

                "prompt":
                    prompt,
            }
        )

    print(
        f"Prompts that could not be extracted: "
        f"{missing_prompt_count}"
    )

    return pd.DataFrame(rows)


# ============================================================
# 11. REMOVE HARD TEST OVERLAP FROM OSS-EVAL
# ============================================================

def remove_hard_overlap_from_oss(
    oss_df,
    hard_df,
):
    """
    Check whether eligible HealthBench Hard prompts are also
    present in OSS-Eval.

    If an eligible OSS example is identical to an eligible Hard
    example, we EXCLUDE THE OSS COPY from train/validation.

    The Hard copy remains eligible for test.

    We do not delete the OSS row. It remains in the manifest
    with:

        included = False

        exclusion_reason =
            "prompt overlaps HealthBench Hard test set"
    """

    oss_df = oss_df.copy()
    hard_df = hard_df.copy()

    # --------------------------------------------------------
    # Eligible Hard prompts only
    # --------------------------------------------------------

    hard_eligible = hard_df[
        hard_df["included"] == True
    ]

    # Ignore missing hashes
    hard_test_hashes = set(
        hard_eligible[
            "prompt_hash"
        ].dropna()
    )

    # --------------------------------------------------------
    # Mark overlap in OSS
    # --------------------------------------------------------

    oss_df[
        "overlaps_hard_test"
    ] = (
        oss_df["prompt_hash"]
        .isin(hard_test_hashes)
    )

    # Hard is the canonical test version.
    hard_df[
        "overlaps_hard_test"
    ] = False

    # --------------------------------------------------------
    # Only currently eligible OSS examples are excluded.
    # --------------------------------------------------------

    overlap_mask = (
        (oss_df["included"] == True)
        &
        (
            oss_df[
                "overlaps_hard_test"
            ] == True
        )
    )

    overlap_count = int(
        overlap_mask.sum()
    )

    # --------------------------------------------------------
    # Print overlap audit BEFORE modifying labels
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HEALTHBENCH HARD OVERLAP AUDIT")
    print("=" * 70)

    print(
        "\nEligible OSS examples overlapping "
        f"eligible Hard examples: {overlap_count}"
    )

    if overlap_count > 0:

        print(
            "\nOverlapping examples by binary label:"
        )

        print(
            oss_df.loc[
                overlap_mask,
                "binary_label"
            ]
            .value_counts()
            .sort_index()
        )

    # --------------------------------------------------------
    # Exclude OSS duplicates
    # --------------------------------------------------------

    oss_df.loc[
        overlap_mask,
        "included"
    ] = False

    oss_df.loc[
        overlap_mask,
        "exclusion_reason"
    ] = (
        "prompt overlaps HealthBench Hard test set"
    )

    return (
        oss_df,
        hard_df,
    )


# ============================================================
# 12. ASSIGN TRAIN / VALIDATION / TEST
# ============================================================

def assign_splits(
    oss_df,
    hard_df,
    seed=42,
):
    """
    Experimental design:

        OSS-Eval after Hard-overlap removal
            -> 80% training
            -> 20% validation

        HealthBench Hard
            -> test

    Stratification preserves the binary class distribution.
    """

    oss_df = oss_df.copy()
    hard_df = hard_df.copy()

    # --------------------------------------------------------
    # Eligible OSS examples after deduplication
    # --------------------------------------------------------

    eligible_oss = oss_df[
        oss_df["included"] == True
    ].copy()

    if len(eligible_oss) == 0:

        raise ValueError(
            "No eligible OSS-Eval examples remain."
        )

    # --------------------------------------------------------
    # Stratified 80/20 split
    # --------------------------------------------------------

    (
        train_indices,
        val_indices,
    ) = train_test_split(

        eligible_oss.index,

        test_size=0.20,

        random_state=seed,

        stratify=eligible_oss[
            "binary_label"
        ],
    )

    oss_df.loc[
        train_indices,
        "split"
    ] = "train"

    oss_df.loc[
        val_indices,
        "split"
    ] = "validation"

    # --------------------------------------------------------
    # Every eligible Hard example becomes test
    # --------------------------------------------------------

    eligible_hard_indices = hard_df[
        hard_df["included"] == True
    ].index

    hard_df.loc[
        eligible_hard_indices,
        "split"
    ] = "test"

    # --------------------------------------------------------
    # Explicitly mark excluded examples
    # --------------------------------------------------------

    oss_df.loc[
        oss_df["included"] == False,
        "split"
    ] = "excluded"

    hard_df.loc[
        hard_df["included"] == False,
        "split"
    ] = "excluded"

    return (
        oss_df,
        hard_df,
    )


# ============================================================
# 13. PRINT MANIFEST SUMMARY
# ============================================================

def print_summary(df):
    """
    Print a complete dataset audit.

    These counts will later be useful for:
        - the revised manuscript
        - reviewer rebuttal
        - supplementary material
    """

    print("\n" + "=" * 70)
    print("MANIFEST SUMMARY")
    print("=" * 70)

    # --------------------------------------------------------
    # Included / excluded
    # --------------------------------------------------------

    print(
        "\nCounts by dataset variant:"
    )

    print(
        df.groupby(
            [
                "dataset_variant",
                "included",
            ],
            dropna=False,
        ).size()
    )

    # --------------------------------------------------------
    # Experimental splits
    # --------------------------------------------------------

    print(
        "\nExperimental split sizes:"
    )

    print(
        df[
            df["included"] == True
        ]["split"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Class counts
    # --------------------------------------------------------

    print(
        "\nClass counts by split:"
    )

    print(
        df[
            df["included"] == True
        ]
        .groupby(
            [
                "split",
                "binary_label",
            ]
        )
        .size()
    )

    # --------------------------------------------------------
    # Class proportions
    # --------------------------------------------------------

    print(
        "\nClass proportions by split:"
    )

    included = df[
        df["included"] == True
    ]

    proportions = (
        included
        .groupby("split")[
            "binary_label"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
    )

    print(proportions)

    # --------------------------------------------------------
    # Clean OSS pool BEFORE train-val split
    # --------------------------------------------------------

    clean_oss = df[
        (
            df["dataset_variant"]
            == "oss_eval"
        )
        &
        (
            df["included"]
            == True
        )
    ]

    print(
        "\nClean OSS-Eval pool after Hard overlap removal:"
    )

    print(
        f"Total: {len(clean_oss)}"
    )

    print(
        clean_oss[
            "binary_label"
        ]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Hard test distribution
    # --------------------------------------------------------

    hard_test = df[
        (
            df["dataset_variant"]
            == "hard"
        )
        &
        (
            df["included"]
            == True
        )
    ]

    print(
        "\nHealthBench Hard test pool:"
    )

    print(
        f"Total: {len(hard_test)}"
    )

    print(
        hard_test[
            "binary_label"
        ]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Original category frequencies
    # --------------------------------------------------------

    print(
        "\nMost common physician-agreed categories:"
    )

    exploded = (
        df[
            "physician_agreed_categories"
        ]
        .fillna("")
        .str.split("|")
        .explode()
    )

    exploded = exploded[
        exploded != ""
    ]

    print(
        exploded
        .value_counts()
        .head(30)
    )

    # --------------------------------------------------------
    # Exclusion reasons
    # --------------------------------------------------------

    print(
        "\nExclusion reasons:"
    )

    print(
        df[
            df["included"] == False
        ][
            "exclusion_reason"
        ]
        .value_counts()
    )

    # --------------------------------------------------------
    # Missing prompts
    # --------------------------------------------------------

    print(
        "\nMissing prompt hashes:"
    )

    print(
        df[
            "prompt_hash"
        ].isna().sum()
    )


# ============================================================
# 14. VALIDATE THE FINAL MANIFEST
# ============================================================

def validate_manifest(df):
    """
    Perform basic safety checks.

    These are not assumptions about the expected paper counts.

    They only check for obvious implementation errors.
    """

    # --------------------------------------------------------
    # Every included example must have a valid binary label
    # --------------------------------------------------------

    included = df[
        df["included"] == True
    ]

    invalid_labels = included[
        ~included[
            "binary_label"
        ].isin([0, 1])
    ]

    if len(invalid_labels) > 0:

        raise ValueError(
            "Some included examples do not "
            "have binary label 0 or 1."
        )

    # --------------------------------------------------------
    # Every included example must have a split
    # --------------------------------------------------------

    if included[
        "split"
    ].isna().any():

        raise ValueError(
            "Some included examples do not "
            "have a train/validation/test split."
        )

    # --------------------------------------------------------
    # No prompt hash should occur in both train and test
    # --------------------------------------------------------

    train_hashes = set(
        df.loc[
            df["split"] == "train",
            "prompt_hash"
        ].dropna()
    )

    test_hashes = set(
        df.loc[
            df["split"] == "test",
            "prompt_hash"
        ].dropna()
    )

    leakage = (
        train_hashes
        & test_hashes
    )

    if leakage:

        raise ValueError(
            f"DATA LEAKAGE DETECTED: "
            f"{len(leakage)} prompt hashes "
            f"occur in both train and test."
        )

    # --------------------------------------------------------
    # Same for validation vs test
    # --------------------------------------------------------

    val_hashes = set(
        df.loc[
            df["split"]
            == "validation",
            "prompt_hash"
        ].dropna()
    )

    leakage = (
        val_hashes
        & test_hashes
    )

    if leakage:

        raise ValueError(
            f"DATA LEAKAGE DETECTED: "
            f"{len(leakage)} prompt hashes "
            f"occur in both validation and test."
        )

    print(
        "\nManifest validation passed."
    )

    print(
        "No train/test or "
        "validation/test prompt overlap detected."
    )


# ============================================================
# 15. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Construct the BRIDGE HealthBench "
            "evidence-sufficiency manifest."
        )
    )

    parser.add_argument(
        "--oss",
        required=True,
        help=(
            "Path to HealthBench "
            "OSS-Eval JSONL file."
        ),
    )

    parser.add_argument(
        "--hard",
        required=True,
        help=(
            "Path to HealthBench "
            "Hard JSONL file."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/manifests/"
            "healthbench_evidence_sufficiency_v1.csv"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed used for the "
            "OSS train/validation split."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # STEP A:
    # Read and label OSS-Eval
    # ========================================================

    oss_df = process_file(
        path=args.oss,
        dataset_variant="oss_eval",
    )

    # ========================================================
    # STEP B:
    # Read and label HealthBench Hard
    # ========================================================

    hard_df = process_file(
        path=args.hard,
        dataset_variant="hard",
    )

    # ========================================================
    # STEP C:
    # Remove Hard-test duplicates from OSS
    # ========================================================

    (
        oss_df,
        hard_df,
    ) = remove_hard_overlap_from_oss(
        oss_df,
        hard_df,
    )

    # ========================================================
    # STEP D:
    # Create train / validation / test splits
    # ========================================================

    (
        oss_df,
        hard_df,
    ) = assign_splits(
        oss_df,
        hard_df,
        seed=args.seed,
    )

    # ========================================================
    # STEP E:
    # Combine everything into ONE master manifest
    # ========================================================

    manifest = pd.concat(
        [
            oss_df,
            hard_df,
        ],
        ignore_index=True,
    )

    # ========================================================
    # STEP F:
    # Validate no leakage remains
    # ========================================================

    validate_manifest(
        manifest
    )

    # ========================================================
    # STEP G:
    # Save manifest
    # ========================================================

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        output_path,
        index=False,
    )

    # ========================================================
    # STEP H:
    # Print audit
    # ========================================================

    print_summary(
        manifest
    )

    print(
        "\nManifest saved to:"
    )

    print(
        output_path.resolve()
    )


if __name__ == "__main__":
    main()