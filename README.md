# BRIDGE

**Detecting Evidence Insufficiency for Gated Hallucination Mitigation in Medical Large Language Models**

Code accompanying the submitted revision to the **IEEE Journal of Biomedical and Health Informatics (JBHI)**.

Paper repository: https://github.com/sankeerthana14/BRIDGE

## Overview

BRIDGE studies **evidence insufficiency** as a pre-generation risk condition in medical large language models (LLMs): cases where the model may possess relevant medical knowledge, but the user prompt does not contain enough patient-specific evidence to support the requested clinical decision.

The paper separates three questions that are often conflated:

1. **Decodability** — is evidence insufficiency linearly accessible from internal representations?
2. **Causal controllability** — does intervening on the decoded representation reliably change model behavior?
3. **Downstream utility** — do those behavioral changes improve the quality of the final response?

The resulting framework treats the evidence-insufficiency probe as a **sensor/router** that determines *when* to intervene, while a separate behavioral mechanism determines *how* generation is changed.

The repository contains code for:

- HealthBench evidence-sufficiency dataset construction and split auditing
- activation extraction for five medical LLMs
- linear probe training and threshold selection
- stronger detection baselines, including SAPLMA-style MLP, SelfCheckGPT, Semantic Entropy, verbalized confidence, token entropy, RoBERTa, TF-IDF, and INSIDE/EigenScore
- label-family sensitivity experiments
- frozen transfer to ClinDet-Bench
- direct CAV intervention experiments
- the adaptive M2 behavioral controller
- HealthBench rubric-based mitigation evaluation
- local-judge validation against HealthBench physician meta-evaluation data
- inference-latency benchmarking

## Paper models

| Model | Backbone | Selected probe layer | Validation threshold | HealthBench Hard AUROC |
|---|---|---:|---:|---:|
| BioMistral-7B | Mistral | 17 | 0.313 | 0.751 |
| Llama3-OpenBioLLM-8B | Llama 3 | 16 | 0.704 | 0.681 |
| UltraMedical-LLaMA3-8B | Llama 3 | 14 | 0.596 | 0.732 |
| MedGemma-1.5-4B-IT | Gemma 3 | 29 | 0.202 | 0.727 |
| Lingshu-7B | Qwen2.5-VL | 22 | 0.304 | 0.741 |

The HealthBench evidence-sufficiency split used in the paper is:

| Split | Sufficient | Insufficient | Total |
|---|---:|---:|---:|
| Train | 310 | 182 | 492 |
| Validation | 78 | 45 | 123 |
| HealthBench Hard test | 54 | 134 | 188 |

External transfer is evaluated on **94 ClinDet-Bench examples** with the HealthBench-trained probes frozen.

## Recommended repository layout

```text
BRIDGE/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── PATHS.example.py
│
├── data/
│   ├── README.md
│   └── manifests/
│       ├── healthbench_manifest.csv
│       └── clindet_manifest.csv
│
├── scripts/
│   ├── download.py
│   ├── prepare_healthbench_manifest.py
│   ├── create_clindet_manifest.py
│   ├── extract_activations.py
│   ├── validate_activations.py
│   ├── train_probes.py
│   ├── evaluate_healthbench_local.py
│   ├── grade_cached_healthbench.py
│   ├── healthbench_label_family_sensitivity.py
│   ├── causal_interventions.py
│   ├── validate_healthbench_local_judge.py
│   ├── mitigation_scripts/
│   └── SLURM_scripts/
│
├── baselines/
│   └── ...
│
├── results/
│   ├── detection/
│   ├── baselines/
│   ├── sensitivity/
│   ├── clindet/
│   ├── causal/
│   ├── mitigation/
│   ├── judge_validation/
│   └── latency/
│
└── figures/
    └── ...
```

The exact directory names do not need to match this layout perfectly, but the public release should clearly separate **source code**, **small reproducibility artifacts**, **paper results**, and **large regenerable caches**.

## Environment

Create a fresh Python environment and install the dependencies used by the project.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA experiments, install the PyTorch build appropriate for the CUDA version on your system.

The experiments require access to the corresponding Hugging Face model checkpoints. Model weights are **not** included in this repository and remain subject to their original licenses and access conditions.

## Path configuration

Do not hard-code cluster-specific paths or personal home directories in the public release.

A recommended pattern is:

```bash
cp PATHS.example.py PATHS.py
```

Then edit `PATHS.py` locally to point to:

- model checkpoints
- downloaded datasets
- cache directories
- output directories

`PATHS.py` should be excluded from Git if it contains machine-specific paths.

## Reproducing the paper

### 1. Prepare the datasets

Use:

```text
scripts/download.py
scripts/prepare_healthbench_manifest.py
scripts/create_clindet_manifest.py
```

The HealthBench manifest records the source row, label family, binary label, inclusion decision, split assignment, prompt hash, and prompt text or source identifier used by the experiment.

The paper maps:

```text
Context-Seeking:
  enough-context                  -> 0 (sufficient)
  not-enough-context              -> 1 (insufficient)

Health-Data-Task:
  enough-info-to-complete-task    -> 0 (sufficient)
  not-enough-info-to-complete-task-> 1 (insufficient)
```

HealthBench Hard is reserved for final test evaluation. No Hard labels are used for fitting the probes, selecting layers, or selecting operating thresholds.

If redistribution of any source dataset is restricted by its license, distribute only the manifest metadata, IDs/hashes, and preparation scripts required to reconstruct the exact split.

### 2. Extract activations

Use:

```text
scripts/extract_activations.py
scripts/validate_activations.py
```

The probe input is the residual representation at the **last prompt token** before response generation.

Activation caches can be large and should normally be regenerated locally rather than committed to Git.

### 3. Train the evidence-insufficiency probes

Use:

```text
scripts/train_probes.py
```

Paper configuration:

```text
classifier:       L2 logistic regression
C:                1
class weighting:  balanced
solver:           liblinear
random seed:      42
max iterations:   10000
standardization:  none
```

A separate probe is fit at every layer using TRAIN. The layer with the highest validation AUROC is selected, ties prefer the lower layer, and the selected probe is refit on TRAIN only.

Threshold selection uses the complete set of validation-score midpoints together with `{0, 1}` and maximizes validation balanced accuracy. Test labels are not used for threshold selection.

### 4. Run detection baselines

Baseline implementations and outputs are under:

```text
baselines/
outputs/baselines/
```

The paper evaluates:

```text
SAPLMA-style hidden-state MLP
SelfCheckGPT-NLI
Semantic Entropy
mean token entropy
constrained verbalized confidence
INSIDE/EigenScore
RoBERTa prompt classifier
TF-IDF logistic regression
```

For INSIDE/EigenScore, the paper uses 10 generated response samples per example and mean hidden representations over generated response tokens at the model's middle hidden layer. EigenScore uses `alpha = 1e-3`, with the operating threshold selected on HealthBench validation only.

### 5. Label-family sensitivity and external transfer

Use:

```text
scripts/healthbench_label_family_sensitivity.py
scripts/create_clindet_manifest.py
```

The label-family experiment repeats probe training, layer selection, threshold selection, and held-out testing independently for Context-Seeking and Health-Data-Task examples.

For ClinDet-Bench, the HealthBench-trained probes are frozen. No ClinDet examples are used for probe fitting, layer selection, or threshold selection.

### 6. Direct CAV intervention

Use:

```text
scripts/causal_interventions.py
```

The direct intervention is performed at the frozen probe layer:

```text
h_tilde = h + epsilon * v_CAV
epsilon in {1, 2, 4, 6, 8, 10}
```

All six strengths are reported. No test-set "best" intervention strength is selected.

### 7. Adaptive M2 controller and selective mitigation

The final M2 and mitigation code is under:

```text
scripts/mitigation_scripts/
```

Final paper configuration:

```text
behavior basis rank: 4
ridge penalty:       10
steering strength:   1
steering layer:      20
```

M2 is learned from the 492 TRAIN examples only.

The mitigation suite compares:

```text
no intervention
always-on safety instruction
direct CAV intervention
probe-gated safety instruction
M2-only intervention
Full BRIDGE = probe-gated instruction + M2
```

Ground-truth HealthBench Hard labels are never used for routing.

### 8. HealthBench rubric evaluation

Use:

```text
scripts/grade_cached_healthbench.py
scripts/validate_healthbench_local_judge.py
```

The paper uses **Qwen2.5-14B-Instruct** as the local HealthBench rubric judge and the official HealthBench `calculate_score` procedure.

Intervention effects are reported as paired candidate-minus-baseline score differences with **2,000 paired bootstrap replicates**, separately for:

```text
all prompts
ground-truth insufficient prompts
ground-truth sufficient prompts
```

These results are **rubric-based surrogate measurements of response quality**, not direct clinician validation of BRIDGE outputs.

The local judge is separately meta-evaluated against the HealthBench physician meta-evaluation data.

### 9. Inference latency

Latency experiments should reproduce the paper protocol:

```text
GPU:                  NVIDIA RTX A6000
prompts per model:    30 HealthBench Hard prompts
warm-up prompts:      2
timing:               CUDA-synchronized wall-clock timing
condition order:      randomized
```

Gate-only time is reported separately from end-to-end generation time because intervention can change response length.

## Small artifacts that should be committed

For reproducibility, commit compact machine-readable files that correspond directly to the paper tables and figures, for example:

```text
results/detection/probe_metrics.csv
results/baselines/baseline_auroc.csv
results/sensitivity/label_family_results.csv
results/clindet/clindet_transfer.csv
results/causal/cav_intervention_summary.csv
results/mitigation/mitigation_summary.csv
results/judge_validation/judge_validation_metrics.csv
results/latency/latency_summary.csv
```

These should contain the final values used in the manuscript.

## Large files that should not be committed

Do **not** commit:

```text
cache/activations/
cache/baseline_generations/
models/
logs/
logs_workstation/
__pycache__/
*.zip
*.bak
.env
API keys or Hugging Face tokens
machine-specific PATHS.py
obsolete/OLD experiment folders
```

If raw generations or activation caches are required for archival reproducibility, store them separately using an archival service or large-file storage and provide a link plus checksums.

## Exact paper-release recommendation

For the JBHI submission, create an immutable Git tag such as:

```bash
git tag -a v1.0-jbhi-submission -m "Code corresponding to submitted JBHI revision"
git push origin v1.0-jbhi-submission
```

This makes it possible to identify the exact code state corresponding to the manuscript even if the repository continues to evolve.

## Reproducibility notes

The main paper intentionally distinguishes between:

```text
decodability
      !=
causal controllability
      !=
downstream utility
```

Accordingly, the direct CAV intervention, M2-only intervention, and downstream mitigation experiments should be reproduced and reported separately rather than selecting only favorable intervention conditions.

The code release should preserve:

- exact train/validation/test split definitions
- random seeds
- selected probe layers and thresholds
- complete threshold-validation sweeps
- all six CAV intervention strengths
- final M2 hyperparameters
- bootstrap procedures
- local-judge configuration
- latency measurement protocol

## Citation

If you use this code, please cite the accompanying manuscript:

```bibtex
@article{satini2026bridge,
  title   = {Detecting Evidence Insufficiency for Gated Hallucination Mitigation in Medical Large Language Models},
  author  = {Satini, Sankeerthana and Tan, Chee Wei},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Submitted revision}
}
```

Please update the citation with the final bibliographic information if the paper is published.

## License

Add a software license before making the repository public. An MIT or Apache-2.0 license is appropriate for many research-code releases, subject to NTU and project-specific requirements.

Dataset and pretrained-model licenses remain those of their original authors and are not superseded by the license for this repository.
