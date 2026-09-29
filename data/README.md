# Data

This directory contains the dataset manifests and reconstruction metadata used for the BRIDGE experiments described in:

**Detecting Evidence Insufficiency for Gated Hallucination Mitigation in Medical Large Language Models**

The paper uses **HealthBench** for training, validation, and held-out evaluation of evidence-insufficiency detection, and **ClinDet-Bench** for frozen external transfer evaluation.

## Directory structure

```text
data/
├── README.md
└── manifests/
    ├── healthbench_evidence_sufficiency_v1.csv
    ├── clindet_manifest.csv
    └── excluded_categories.txt
```

Additional raw benchmark files may be kept locally for experiment execution, but the manifests above define the exact data used by BRIDGE.

---

## 1. HealthBench evidence-sufficiency dataset

BRIDGE operationalizes **input evidence sufficiency** using two physician-agreed HealthBench annotation families.

### Context-Seeking

| HealthBench category | BRIDGE label |
|---|---:|
| `enough-context` | 0 = sufficient evidence |
| `not-enough-context` | 1 = insufficient evidence |

### Health-Data-Task

| HealthBench category | BRIDGE label |
|---|---:|
| `enough-info-to-complete-task` | 0 = sufficient evidence |
| `not-enough-info-to-complete-task` | 1 = insufficient evidence |

The paper deliberately excludes broader uncertainty/context categories so that the binary task does not conflate missing patient-specific evidence with other forms of uncertainty.

HealthBench was **not** designed as a dedicated evidence-insufficiency benchmark. Accordingly, these labels should be interpreted as an **operationalization of evidence sufficiency**, not as independent clinical ground truth for hallucination.

### Final split used in the paper

| Split | Sufficient | Insufficient | Total |
|---|---:|---:|---:|
| Train | 310 | 182 | 492 |
| Validation | 78 | 45 | 123 |
| HealthBench Hard test | 54 | 134 | 188 |

The resulting 803 included examples comprise:

- 408 Context-Seeking examples
- 395 Health-Data-Task examples

Across the 6,000 OSS-Eval and Hard records audited in the paper:

- 5,009 examples did not belong to one of the target label families
- 188 OSS-Eval prompts overlapped the reserved HealthBench Hard test set and were excluded from train/validation eligibility

The HealthBench Hard split is reserved for final evaluation and is **not** used for:

- probe fitting
- layer selection
- threshold selection
- model selection

### HealthBench manifest

The exact construction is recorded in:

```text
manifests/healthbench_evidence_sufficiency_v1.csv
```

The paper states that the immutable manifest records:

- source row
- HealthBench category
- BRIDGE binary label
- inclusion decision
- split assignment
- prompt hash
- prompt text

Prompt hashes are used to audit overlap between training/validation data and the reserved HealthBench Hard test set.

The script used to reconstruct this manifest is:

```text
../scripts/prepare_healthbench_manifest.py
```

---

## 2. ClinDet-Bench external transfer set

ClinDet-Bench is used only for **external transfer evaluation** of the frozen HealthBench-trained probes.

The paper evaluates 94 ClinDet-Bench examples:

| Original ClinDet label | Count | BRIDGE label |
|---|---:|---:|
| Complete | 32 | 0 = sufficient evidence |
| Incomplete-Determinable | 30 | 0 = sufficient evidence |
| Incomplete-Undeterminable | 32 | 1 = insufficient evidence |
| **Total** | **94** | |

The mapping follows **clinical determinability**:

- missing information is not automatically treated as insufficient;
- an example is labeled insufficient only when the missing information prevents the requested clinical judgment.

The exact external-evaluation set is recorded in:

```text
manifests/clindet_manifest.csv
```

The corresponding construction script is:

```text
../scripts/create_clindet_manifest.py
```

### Important evaluation rule

ClinDet-Bench is not used for any training or model selection.

For the external transfer experiment:

- the HealthBench-trained probe is frozen;
- the selected probe layer is frozen;
- the HealthBench validation threshold is frozen;
- no ClinDet example is used for retraining, layer reselection, or threshold reselection.

This experiment therefore measures out-of-distribution transfer of the HealthBench-trained detector.

---

## 3. Label-family sensitivity experiment

To test whether the main HealthBench result depends on pooling heterogeneous annotation families, the paper repeats the complete pipeline independently for:

```text
Context-Seeking
Health-Data-Task
```

For each family, the following are repeated independently while preserving the original split assignments:

1. probe training
2. layer selection
3. threshold selection
4. held-out testing

This analysis is implemented in:

```text
../scripts/healthbench_label_family_sensitivity.py
```

---

## 4. Excluded categories

The file:

```text
manifests/excluded_categories.txt
```

documents HealthBench categories excluded from the evidence-sufficiency task.

These exclusions are intentional and are used to avoid conflating evidence insufficiency with broader uncertainty or unrelated annotation constructs.

---

## 5. Reconstructing the data

A typical reconstruction workflow is:

```text
official HealthBench release
        ↓
prepare_healthbench_manifest.py
        ↓
healthbench_evidence_sufficiency_v1.csv
        ↓
BRIDGE train / validation / Hard test experiments
```

and:

```text
official ClinDet-Bench release
        ↓
create_clindet_manifest.py
        ↓
clindet_manifest.csv
        ↓
frozen external-transfer evaluation
```

Users should obtain the upstream datasets from their official sources and comply with the original licenses and terms of use.

---

## 6. Reproducibility notes

The manifests are intended to make the paper's dataset construction auditable and reproducible.

For exact reproduction, preserve:

- the original split assignments
- the binary label mapping above
- prompt-hash overlap checks
- the reserved HealthBench Hard test set
- the 94-example ClinDet external evaluation set
- the rule that ClinDet does not influence probe fitting or model selection

The repository's main `README.md` contains the full experiment-reproduction instructions.
