import pandas as pd

manifest = pd.read_csv(
    "data/manifests/healthbench_evidence_sufficiency_v1.csv"
)

excluded = manifest[
    manifest["exclusion_reason"] == "no target evidence-sufficiency category"
].copy()

# One row can have multiple physician-agreed categories,
# separated by "|", so split and explode them.
excluded_categories = (
    excluded["physician_agreed_categories"]
    .fillna("")
    .str.split("|")
    .explode()
)

excluded_categories = excluded_categories[
    excluded_categories != ""
]

print(
    excluded_categories
    .value_counts()
    .head(50)
)