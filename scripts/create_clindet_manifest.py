# -*- coding: utf-8 -*-

"""
Create immutable ClinDet evidence-sufficiency manifest.

Canonical inputs
----------------
Benchmark:
    data/clindet/benchmark/Clinical_Decision_Task.xlsx

Optional/fallback result file:
    data/clindet/results/Clinical_Decision_Task_Result_base.xlsx

Output:
    data/manifests/clindet_evidence_sufficiency_v1.csv

Binary mapping
--------------
Complete                  -> 0  sufficient
Incomplete_Determinable   -> 0  sufficient
Incomplete_Undeterminable -> 1  insufficient

Expected dataset
----------------
Total                      94
Complete                   32
Incomplete_Determinable    30
Incomplete_Undeterminable  32

Binary:
y=0                        62
y=1                        32

The script also cross-checks all labels and example IDs
against the already validated BRIDGE ClinDet activation cache.
"""

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ============================================================
# EXPECTED STRUCTURE
# ============================================================

EXPECTED_TOTAL = 94

EXPECTED_SUBGROUP_COUNTS = {
    "Complete": 32,
    "Incomplete_Determinable": 30,
    "Incomplete_Undeterminable": 32,
}

EXPECTED_LABEL_COUNTS = {
    0: 62,
    1: 32,
}

LABEL_MAP = {
    "Complete": 0,
    "Incomplete_Determinable": 0,
    "Incomplete_Undeterminable": 1,
}


# ============================================================
# POSSIBLE FIELD NAMES
# ============================================================

PROMPT_COLUMNS = [
    "clinical_decision_task",
    "clinical decision task",
    "decision_task",
    "decision task",
    "prompt",
    "question",
    "scenario",
    "case",
    "clinical_case",
    "clinical case",
    "input",
    "text",
    "task",
]

CONDITION_COLUMNS = [
    "information_condition",
    "information condition",
    "condition",
    "completeness",
    "category",
    "subgroup",
    "type",
    "information_status",
    "information status",
]

ID_COLUMNS = [
    "id",
    "case_id",
    "case id",
    "example_id",
    "example id",
    "index",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_name(value):

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def normalize_condition(value):

    value = str(value).strip()

    key = normalize_name(value)

    aliases = {
        "complete":
            "Complete",

        "incomplete_determinable":
            "Incomplete_Determinable",

        "incompletedeterminable":
            "Incomplete_Determinable",

        "incomplete_undeterminable":
            "Incomplete_Undeterminable",

        "incompleteundeterminable":
            "Incomplete_Undeterminable",
    }

    return aliases.get(
        key,
        value,
    )


def prompt_hash(text):

    normalized = " ".join(
        str(text).strip().split()
    )

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def find_column(
    df,
    candidates,
):

    column_map = {
        normalize_name(column): column
        for column in df.columns
    }

    for candidate in candidates:

        key = normalize_name(
            candidate
        )

        if key in column_map:

            return column_map[
                key
            ]

    return None


# ============================================================
# INSPECT EXCEL WORKBOOK
# ============================================================

def inspect_workbook(
    path,
    title,
):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"{title} not found:\n{path}"
        )

    workbook = pd.ExcelFile(
        path,
        engine="openpyxl",
    )

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print("Path:", path)
    print("Sheets:")
    print(workbook.sheet_names)

    sheets = {}

    for sheet_name in workbook.sheet_names:

        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
            engine="openpyxl",
        )

        # Remove fully empty rows/columns
        df = df.dropna(
            axis=0,
            how="all",
        )

        df = df.dropna(
            axis=1,
            how="all",
        )

        df = df.reset_index(
            drop=True
        )

        sheets[
            sheet_name
        ] = df

        print()
        print(
            f"Sheet: {sheet_name}"
        )

        print(
            f"Shape: {df.shape}"
        )

        print(
            "Columns:",
            list(df.columns),
        )

        if len(df) > 0:

            print(
                df.head(2).to_string(
                    index=False
                )
            )

    return sheets


# ============================================================
# EXTRACT RECORDS FROM A SHEET
# ============================================================

def extract_records_from_sheet(
    df,
    sheet_name,
    source_path,
):

    if len(df) == 0:

        return []

    prompt_col = find_column(
        df,
        PROMPT_COLUMNS,
    )

    condition_col = find_column(
        df,
        CONDITION_COLUMNS,
    )

    id_col = find_column(
        df,
        ID_COLUMNS,
    )

    # --------------------------------------------------------
    # Condition may be encoded in sheet name
    # --------------------------------------------------------

    sheet_condition = (
        normalize_condition(
            sheet_name
        )
    )

    if (
        sheet_condition
        not in LABEL_MAP
    ):

        sheet_condition = None

    # --------------------------------------------------------
    # Must have prompt
    # --------------------------------------------------------

    if prompt_col is None:

        return []

    # Must get condition either from column or sheet name
    if (
        condition_col is None
        and
        sheet_condition is None
    ):

        return []

    records = []

    for row_idx, row in df.iterrows():

        prompt_value = row[
            prompt_col
        ]

        if pd.isna(
            prompt_value
        ):

            continue

        prompt = str(
            prompt_value
        ).strip()

        if not prompt:

            continue

        # Determine subgroup
        if condition_col is not None:

            condition_value = row[
                condition_col
            ]

            if pd.isna(
                condition_value
            ):

                if sheet_condition is None:

                    continue

                subgroup = (
                    sheet_condition
                )

            else:

                subgroup = (
                    normalize_condition(
                        condition_value
                    )
                )

        else:

            subgroup = (
                sheet_condition
            )

        if subgroup not in LABEL_MAP:

            continue

        # Optional original ID
        source_id = None

        if id_col is not None:

            value = row[
                id_col
            ]

            if not pd.isna(
                value
            ):

                source_id = str(
                    value
                ).strip()

        records.append(
            {
                "prompt":
                    prompt,

                "subgroup":
                    subgroup,

                "label":
                    LABEL_MAP[
                        subgroup
                    ],

                "source_sheet":
                    str(
                        sheet_name
                    ),

                "source_row":
                    int(
                        row_idx
                    ),

                "source_id":
                    source_id,

                "source_file":
                    str(
                        source_path
                    ),

                "prompt_sha256":
                    prompt_hash(
                        prompt
                    ),
            }
        )

    return records


# ============================================================
# EXTRACT FROM WORKBOOK
# ============================================================

def extract_from_workbook(
    workbook_path,
    workbook_sheets,
):

    records = []

    for (
        sheet_name,
        df,
    ) in workbook_sheets.items():

        sheet_records = (
            extract_records_from_sheet(
                df=df,
                sheet_name=sheet_name,
                source_path=workbook_path,
            )
        )

        records.extend(
            sheet_records
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# TRY WIDE-TABLE STRUCTURE
# ============================================================

def extract_wide_structure(
    sheets,
    source_path,
):

    """
    Some benchmark workbooks may have columns such as:

        Complete
        Incomplete_Determinable
        Incomplete_Undeterminable

    rather than a single condition column.

    This function handles that format.
    """

    rows = []

    subgroup_columns = {
        "Complete":
            None,

        "Incomplete_Determinable":
            None,

        "Incomplete_Undeterminable":
            None,
    }

    for sheet_name, df in (
        sheets.items()
    ):

        normalized_columns = {
            normalize_name(
                c
            ): c
            for c
            in df.columns
        }

        found = {}

        for subgroup in (
            subgroup_columns.keys()
        ):

            key = normalize_name(
                subgroup
            )

            if key in normalized_columns:

                found[
                    subgroup
                ] = normalized_columns[
                    key
                ]

        if len(found) < 1:

            continue

        print()
        print(
            "Detected possible wide-form "
            f"ClinDet sheet: {sheet_name}"
        )

        print(
            "Condition columns:",
            found,
        )

        for subgroup, column in (
            found.items()
        ):

            for row_idx, value in (
                df[
                    column
                ].items()
            ):

                if pd.isna(
                    value
                ):

                    continue

                prompt = str(
                    value
                ).strip()

                if not prompt:

                    continue

                rows.append(
                    {
                        "prompt":
                            prompt,

                        "subgroup":
                            subgroup,

                        "label":
                            LABEL_MAP[
                                subgroup
                            ],

                        "source_sheet":
                            sheet_name,

                        "source_row":
                            int(
                                row_idx
                            ),

                        "source_id":
                            None,

                        "source_file":
                            str(
                                source_path
                            ),

                        "prompt_sha256":
                            prompt_hash(
                                prompt
                            ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CHOOSE DATA SOURCE
# ============================================================

def build_source_records(
    benchmark_path,
    result_path,
):

    benchmark_sheets = (
        inspect_workbook(
            benchmark_path,
            "CLINDET BENCHMARK WORKBOOK",
        )
    )

    result_sheets = (
        inspect_workbook(
            result_path,
            "CLINDET BASE-RESULT WORKBOOK",
        )
    )

    candidates = []

    # --------------------------------------------------------
    # Benchmark normal extraction
    # --------------------------------------------------------

    benchmark_records = (
        extract_from_workbook(
            benchmark_path,
            benchmark_sheets,
        )
    )

    if len(
        benchmark_records
    ) > 0:

        candidates.append(
            (
                "benchmark_normal",
                benchmark_records,
            )
        )

    # --------------------------------------------------------
    # Benchmark wide extraction
    # --------------------------------------------------------

    benchmark_wide = (
        extract_wide_structure(
            benchmark_sheets,
            benchmark_path,
        )
    )

    if len(
        benchmark_wide
    ) > 0:

        candidates.append(
            (
                "benchmark_wide",
                benchmark_wide,
            )
        )

    # --------------------------------------------------------
    # Result-file normal extraction
    # --------------------------------------------------------

    result_records = (
        extract_from_workbook(
            result_path,
            result_sheets,
        )
    )

    if len(
        result_records
    ) > 0:

        candidates.append(
            (
                "result_normal",
                result_records,
            )
        )

    # --------------------------------------------------------
    # Result-file wide extraction
    # --------------------------------------------------------

    result_wide = (
        extract_wide_structure(
            result_sheets,
            result_path,
        )
    )

    if len(
        result_wide
    ) > 0:

        candidates.append(
            (
                "result_wide",
                result_wide,
            )
        )

    print()
    print("=" * 78)
    print("CANDIDATE EXTRACTIONS")
    print("=" * 78)

    valid = []

    for (
        name,
        df,
    ) in candidates:

        counts = (
            df[
                "subgroup"
            ]
            .value_counts()
            .to_dict()
        )

        print()
        print(name)
        print("Rows:", len(df))
        print("Subgroup counts:", counts)

        exact_counts = all(
            counts.get(
                subgroup,
                0,
            )
            ==
            expected
            for (
                subgroup,
                expected
            )
            in EXPECTED_SUBGROUP_COUNTS.items()
        )

        if (
            len(df) == EXPECTED_TOTAL
            and
            exact_counts
        ):

            valid.append(
                (
                    name,
                    df,
                )
            )

    if not valid:

        raise RuntimeError(
            "\nCould not automatically extract "
            "exactly 94 ClinDet records with subgroup "
            "counts 32/30/32 from the two Excel "
            "workbooks.\n\n"
            "The workbook schema has now been printed "
            "above. Send me that output and we can map "
            "the exact columns."
        )

    # Prefer benchmark source if valid
    valid = sorted(
        valid,
        key=lambda x: (
            0
            if x[0].startswith(
                "benchmark"
            )
            else 1
        ),
    )

    selected_name, selected_df = (
        valid[
            0
        ]
    )

    print()
    print(
        "Selected extraction:",
        selected_name,
    )

    return (
        selected_df.reset_index(
            drop=True
        ),
        selected_name,
    )


# ============================================================
# ACTIVATION REFERENCE
# ============================================================

def load_activation_reference(
    activation_root,
    reference_model,
):

    root = (
        Path(
            activation_root
        )
        / reference_model
        / "clindet"
        / "external_test"
    )

    files = sorted(
        root.glob(
            "*.pt"
        )
    )

    if len(
        files
    ) != EXPECTED_TOTAL:

        raise RuntimeError(
            f"Expected 94 ClinDet activation "
            f"files for {reference_model}; "
            f"found {len(files)}."
        )

    rows = []

    pattern = re.compile(
        r"^clindet_(\d+)_"
    )

    for path in files:

        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        example_id = str(
            obj[
                "example_id"
            ]
        )

        label = int(
            obj[
                "label"
            ]
        )

        match = pattern.match(
            example_id
        )

        if match is None:

            raise RuntimeError(
                f"Could not parse source index "
                f"from example ID: {example_id}"
            )

        source_index = int(
            match.group(
                1
            )
        )

        rows.append(
            {
                "example_id":
                    example_id,

                "source_index":
                    source_index,

                "activation_label":
                    label,
            }
        )

    df = pd.DataFrame(
        rows
    ).sort_values(
        "source_index"
    ).reset_index(
        drop=True
    )

    expected_indices = list(
        range(
            EXPECTED_TOTAL
        )
    )

    observed_indices = (
        df[
            "source_index"
        ].tolist()
    )

    if (
        observed_indices
        != expected_indices
    ):

        raise RuntimeError(
            "Activation IDs are not a "
            "complete 0..93 sequence."
        )

    counts = (
        df[
            "activation_label"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if counts != EXPECTED_LABEL_COUNTS:

        raise RuntimeError(
            f"Activation label counts mismatch: "
            f"{counts}"
        )

    print()
    print("=" * 78)
    print("ACTIVATION REFERENCE")
    print("=" * 78)

    print(
        f"Model: {reference_model}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Labels: {counts}"
    )

    return df


# ============================================================
# ALIGN WITH ACTIVATION CACHE
# ============================================================

def align_manifest(
    source_df,
    activation_df,
    extraction_name,
):

    if len(
        source_df
    ) != EXPECTED_TOTAL:

        raise RuntimeError(
            "Source extraction must contain "
            "exactly 94 rows."
        )

    source_df = (
        source_df.copy()
    )

    source_df[
        "source_index"
    ] = np.arange(
        EXPECTED_TOTAL,
        dtype=np.int64,
    )

    merged = (
        activation_df.merge(
            source_df,
            on="source_index",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(
        merged
    ) != EXPECTED_TOTAL:

        raise RuntimeError(
            "Activation/source alignment failed."
        )

    mismatch = (
        merged[
            "activation_label"
        ]
        !=
        merged[
            "label"
        ]
    )

    if mismatch.any():

        print()
        print(
            "WARNING: direct row-order label "
            "alignment does NOT match."
        )

        bad = merged.loc[
            mismatch,
            [
                "source_index",
                "example_id",
                "activation_label",
                "label",
                "subgroup",
            ],
        ]

        print()
        print(
            bad.head(
                30
            ).to_string(
                index=False
            )
        )

        raise RuntimeError(
            "\nClinDet workbook row ordering "
            "does not match the ordering used "
            "when the existing activation cache "
            "was generated.\n\n"
            "Do NOT force this alignment. "
            "Send me this mismatch output; "
            "we will recover the exact original "
            "alignment key."
        )

    manifest = pd.DataFrame(
        {
            "example_id":
                merged[
                    "example_id"
                ],

            "source_index":
                merged[
                    "source_index"
                ],

            "prompt":
                merged[
                    "prompt"
                ],

            "label":
                merged[
                    "label"
                ],

            "subgroup":
                merged[
                    "subgroup"
                ],

            "split":
                "external_test",

            "dataset":
                "clindet",

            "prompt_sha256":
                merged[
                    "prompt_sha256"
                ],

            "source_file":
                merged[
                    "source_file"
                ],

            "source_sheet":
                merged[
                    "source_sheet"
                ],

            "source_row":
                merged[
                    "source_row"
                ],

            "source_id":
                merged[
                    "source_id"
                ],

            "extraction_method":
                extraction_name,
        }
    )

    return manifest


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_manifest(
    df,
):

    if len(
        df
    ) != EXPECTED_TOTAL:

        raise RuntimeError(
            f"Expected 94 rows, "
            f"found {len(df)}."
        )

    if df[
        "example_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate example IDs."
        )

    if df[
        "prompt"
    ].isna().any():

        raise RuntimeError(
            "Missing prompts."
        )

    if (
        df[
            "prompt"
        ]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):

        raise RuntimeError(
            "Empty prompts."
        )

    label_counts = (
        df[
            "label"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    subgroup_counts = (
        df[
            "subgroup"
        ]
        .value_counts()
        .to_dict()
    )

    if label_counts != EXPECTED_LABEL_COUNTS:

        raise RuntimeError(
            f"Label mismatch: "
            f"{label_counts}"
        )

    for (
        subgroup,
        expected,
    ) in (
        EXPECTED_SUBGROUP_COUNTS.items()
    ):

        actual = (
            subgroup_counts.get(
                subgroup,
                0,
            )
        )

        if actual != expected:

            raise RuntimeError(
                f"{subgroup}: expected "
                f"{expected}, found {actual}."
            )

    print()
    print("=" * 78)
    print("FINAL MANIFEST VALIDATION")
    print("=" * 78)

    print(
        "Rows:",
        len(
            df
        ),
    )

    print(
        "Binary label counts:",
        label_counts,
    )

    print(
        "Subgroup counts:",
        subgroup_counts,
    )

    print()
    print(
        "First 10 rows:"
    )

    print(
        df[
            [
                "example_id",
                "source_index",
                "label",
                "subgroup",
                "source_sheet",
            ]
        ]
        .head(
            10
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "FINAL VALIDATION: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark",
        default=(
            "data/clindet/benchmark/"
            "Clinical_Decision_Task.xlsx"
        ),
    )

    parser.add_argument(
        "--base-results",
        default=(
            "data/clindet/results/"
            "Clinical_Decision_Task_Result_base.xlsx"
        ),
    )

    parser.add_argument(
        "--activation-root",
        default="cache/activations",
    )

    parser.add_argument(
        "--reference-model",
        default="biomistral",
    )

    parser.add_argument(
        "--output",
        default=(
            "data/manifests/"
            "clindet_evidence_sufficiency_v1.csv"
        ),
    )

    args = parser.parse_args()

    print("=" * 78)
    print(
        "CLINDET MANIFEST CREATION"
    )
    print("=" * 78)

    print(
        "Benchmark:",
        args.benchmark,
    )

    print(
        "Base results:",
        args.base_results,
    )

    print(
        "Activation root:",
        args.activation_root,
    )

    print(
        "Reference model:",
        args.reference_model,
    )

    print(
        "Output:",
        args.output,
    )

    # --------------------------------------------------------
    # Extract source records
    # --------------------------------------------------------

    (
        source_df,
        extraction_name,
    ) = build_source_records(
        benchmark_path=
            args.benchmark,

        result_path=
            args.base_results,
    )

    # --------------------------------------------------------
    # Existing activation reference
    # --------------------------------------------------------

    activation_df = (
        load_activation_reference(
            activation_root=
                args.activation_root,

            reference_model=
                args.reference_model,
        )
    )

    # --------------------------------------------------------
    # Align
    # --------------------------------------------------------

    manifest = align_manifest(
        source_df=
            source_df,

        activation_df=
            activation_df,

        extraction_name=
            extraction_name,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_manifest(
        manifest
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    print()
    print("=" * 78)
    print(
        "CLINDET MANIFEST CREATED"
    )
    print("=" * 78)

    print(
        output_path.resolve()
    )


if __name__ == "__main__":

    main()