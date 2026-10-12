# Phase 06 Execution Plan: Stochastic Symptom-Reporting Noise

**Project Title**: *Disease Prediction System Using Machine Learning: A Robust, Explainable, and Uncertainty-Aware Framework Under Real-World Symptom Sparsity and Noise*  
**Author**: Nasir uddin (Registration No: `23880300014`, Session: `2023-2024`)  
**Supervisor**: Husne Ara Rubaiyeat, Assistant Professor  
**Department**: Department of Computer Science, National University, Gazipur, Bangladesh  
**Phase Target**: Phase 06 — Stochastic Symptom-Reporting Noise (Experiment E6, Testing Hypothesis H3)  
**Status**: Proposal & Execution Blueprint (Pending Approval)  

---

## 1. Executive Summary & Scientific Motivation

In real-world clinical intake, patients do not merely omit symptoms (sparsity, investigated in Phase 05); they also report inaccurate symptoms:
1. **Symptom Omission / False Negatives ($1 \to 0$)**: An active clinical manifestation is unmentioned, forgotten, or suppressed by the patient (e.g., mild fever omitted).
2. **Erroneous Reporting / False Positives ($0 \to 1$)**: A symptom that is pathologically absent is erroneously endorsed due to anxiety, misunderstanding of medical terminology, or psychosomatic misattribution (e.g., normal fatigue reported as severe lethargy).
3. **Combined Misreporting**: Simultaneous omissions and spurious symptom activations occurring in the same patient intake.

### Central Research Question & Hypothesis
* **RQ3**: *How robust are different machine-learning models to false-positive and false-negative symptom reporting under controlled perturbation levels?*
* **Hypothesis H3**: *Increasing symptom-reporting noise will degrade predictive accuracy, with false-positive symptom activations causing steeper degradation than symptom dropouts (false negatives).*

---

## 2. Critical Methodological Audit: The Denominator Asymmetry Problem

### The Problem
In the Kaggle 132-symptom benchmark:
* Each patient presents an average of **$7.45$ positive symptoms** ($1$s).
* Each patient has an average of **$124.55$ absent symptoms** ($0$s).

Because the denominators differ by more than **16-fold**, applying raw percentages creates an extreme discrepancy in the actual number of perturbed features:
* A **5% dropout rate** alters only $7.45 \times 0.05 \approx \mathbf{0.37 \text{ features}}$ on average.
* A **1% false-positive rate** alters $124.55 \times 0.01 \approx \mathbf{1.25 \text{ features}}$ on average (over $3.3\times$ more feature alterations than 5% dropout!).

### The Solution: Two-Tier Experimental Design
1. **Primary Preregistered Grid (Preregistered Rates for H3)**:
   * Dropout ($p_{\text{drop}}$): $5\%, 10\%, 15\%, 20\%, 30\%$
   * False Positives ($p_{\text{fp}}$): $1\%, 2\%, 5\%, 10\%$
   * Combined: All $5 \times 4 = 20$ grid pairs
2. **Supplementary Matched-Error-Burden Sensitivity Analysis (Fair Scientific Control)**:
   * Match the expected number of altered features between dropout and activation:
     | Dropout Rate ($p_{\text{drop}}$) | Expected Altered Features | Matched FP Rate ($p_{\text{fp}}^{\text{matched}}$) |
     | :---: | :---: | :---: |
     | **5%** | 0.37 symptoms | **0.30%** |
     | **10%** | 0.75 symptoms | **0.60%** |
     | **15%** | 1.12 symptoms | **0.90%** |
     | **20%** | 1.49 symptoms | **1.20%** |
     | **30%** | 2.24 symptoms | **1.79%** |

This guarantees that any observed excess sensitivity to false positives is an intrinsic architectural vulnerability, rather than a mathematical artifact of denominator imbalance.

---

## 3. Experimental Architecture & Hyperparameters

```
┌─────────────────────────────────┐      ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│  Phase 06 Pipeline Ingestion    │ ───► │  Protocol C (10-Fold Groups)    │ ───► │  Model Training on Clean Train  │
│  - Training.csv (4,920 rows)    │      │  - 304 unique symptom templates │      │  - 6 model architectures        │
│  - Testing.csv excluded         │      │  - Zero vector pattern leakage  │      │  - Full 132 & Fold-Isolated 50  │
└─────────────────────────────────┘      └─────────────────────────────────┘      └─────────────────────────────────┘
                                                                                                │
                                                                                                ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│  Artifact & Thesis Freezing     │ ◄─── │  Hierarchical Metric Aggregation│ ◄─── │  Nested Validation Noise Engine │
│  - 300 DPI figures to Book/     │      │  - Stage 1: Seed average / fold │      │  - Scenario A: 5 Dropouts       │
│  - phase6_noise_report.md       │      │  - Stage 2: 10-Fold mean +- SD  │      │  - Scenario B: 4 False Positives│
│  - Consistency overview upgrade │      │  - Macro-F1, Acc, Top-3/5, BS   │      │  - Scenario C: 20 Combined      │
│  - Chapter VII / manuscript     │      │  - Matched-burden supplement    │      │  - 5 Matched-burden controls    │
└─────────────────────────────────┘      └─────────────────────────────────┘      └─────────────────────────────────┘
```

### Frozen Experimental Parameters
* **Dataset**: `Training.csv` (4,920 samples, 41 classes). `Testing.csv` is excluded due to documented 97.6% leakage.
* **Validation**: Protocol C (Duplicate-Aware Group 10-Fold CV; identical symptom vectors isolated in single folds).
* **Models (6)**:
  1. `Logistic Regression`: $L_2$ regularization, $C=1.0$, `max_iter=500`, `random_state=42`.
  2. `Random Forest`: 100 trees, `random_state=42`, `n_jobs=-1`.
  3. `SVM (RBF)`: Radial basis function kernel, probability estimates enabled, `random_state=42`.
  4. `MLP`: Hidden layers (128, 64), early stopping, `random_state=42`.
  5. `XGBoost`: 60 estimators, `max_depth=4`, `random_state=42`, `n_jobs=-1`, `verbosity=0`.
  6. `Multinomial Naive Bayes`: Laplace smoothing $\alpha=1.0$.
* **Representations (2)**:
  1. `Full_132`: Full 132 symptom dimensions.
  2. `Fold_50`: Top 50 features selected strictly inside each training fold using Mutual Information on $X_{\text{train}}, y_{\text{train}}$ (zero validation leakage).
* **Stochastic Perturbation Seeds (5)**: `42`, `101`, `2024`, `777`, `999`.
* **Inference-Only Noise**: Models are trained on clean $X_{\text{train}}$; noise is injected strictly into validation inputs $X_{\text{val}}$.

---

## 4. Mathematical Verification of Output Row Counts

### 1. Primary Preregistered Grid:
* **Clean Baseline**: $10 \text{ folds} \times 2 \text{ reps} \times 6 \text{ models} = 120$ rows
* **Noisy Conditions (29)**:
  * Scenario A (Dropout only, 5 rates): 5
  * Scenario B (False-positive only, 4 rates): 4
  * Scenario C (Combined, $5 \times 4$ pairs): 20
  * Subtotal conditions: $29$
* **Noisy Rows**: $29 \text{ conditions} \times 10 \text{ folds} \times 2 \text{ reps} \times 6 \text{ models} \times 5 \text{ seeds} = 17{,}400$ rows
* **Primary Subtotal**: $120 + 17{,}400 = \mathbf{17{,}520 \text{ rows}}$

### 2. Supplementary Matched-Error-Burden Grid:
* **Matched FP Conditions (5 rates)**: $5 \times 10 \text{ folds} \times 2 \text{ reps} \times 6 \text{ models} \times 5 \text{ seeds} = \mathbf{3{,}000 \text{ rows}}$

### 3. Grand Total Raw Rows:
$$N_{\text{total}} = 17{,}520 + 3{,}000 = \mathbf{20{,}520 \text{ rows}}$$

### Summary Aggregation Rows:
$$N_{\text{summary}} = (1 \text{ baseline} + 29 \text{ primary} + 5 \text{ matched}) \times 2 \text{ reps} \times 6 \text{ models} = 35 \times 12 = \mathbf{420 \text{ rows}}$$

---

## 5. Noise Injection Engine & Nested Invariant

The core generator enforces nested monotonicity via precomputed uniform random matrices:

```python
def apply_reporting_noise(
    X: np.ndarray,
    dropout_uniforms: np.ndarray,
    false_positive_uniforms: np.ndarray,
    p_dropout: float = 0.0,
    p_false_positive: float = 0.0,
) -> np.ndarray:
    """
    Injects stochastic reporting noise into binary symptom vectors.
    1 -> 0 (Dropout): Present symptom omitted if uniform < p_dropout.
    0 -> 1 (Activation): Absent symptom activated if uniform < p_false_positive.
    
    Nested Property: Because uniforms are fixed per record and seed:
    Increasing p_dropout strictly expands omitted symptoms.
    Increasing p_false_positive strictly expands activated symptoms.
    """
    noisy = X.copy()
    present_mask = (X == 1)
    absent_mask = (X == 0)
    
    if p_dropout > 0.0:
        dropped = present_mask & (dropout_uniforms < p_dropout)
        noisy[dropped] = 0.0
        
    if p_false_positive > 0.0:
        activated = absent_mask & (false_positive_uniforms < p_false_positive)
        noisy[activated] = 1.0
        
    return noisy
```

---

## 6. Verification & Automated Test Suite

Before running the full benchmark, implement `research_work/tests/test_noise_experiment.py` asserting:

1. **False-Negative Invariant**: When $p_{\text{fp}} = 0$, only original 1s can become 0; all original 0s remain 0.
2. **False-Positive Invariant**: When $p_{\text{drop}} = 0$, only original 0s can become 1; all original 1s remain 1.
3. **Combined Invariant**: Both mutations occur; output remains strictly binary $\{0, 1\}$.
4. **Nested Monotonicity**: For the same seed, indices dropped at $p=0.10$ must be a strict subset of indices dropped at $p=0.20$.
5. **Array Immutability**: The original clean array is never modified in-place.
6. **Zero Leakage**: Unique symptom pattern hashes in Protocol C do not cross train/val boundaries.
7. **Ranking Monotonicity**: Top-5 $\ge$ Top-3 $\ge$ Top-1 for all predictions.
8. **Probability Sanity**: Output probability distributions sum to $1.0$ across 41 classes with no NaNs.

---

## 7. Deliverables & Outputs

1. **Source Code**:
   - `research_work/src/noise_experiment.py`
   - `research_work/tests/test_noise_experiment.py`
2. **Numerical Datasets**:
   - `research_work/reports/noise_fold_results.csv` (20,520 rows)
   - `research_work/reports/noise_summary.csv` (420 rows)
3. **Publication-Quality Figures (300 DPI)**:
   - `research_work/reports/figures/noise_degradation_curves.png` & `Book_Writing/figures/noise_degradation_curves.png`
   - `research_work/reports/figures/noise_type_comparison.png` & `Book_Writing/figures/noise_type_comparison.png`
   - `research_work/reports/figures/noise_combined_heatmap.png` & `Book_Writing/figures/noise_combined_heatmap.png`
4. **Reports & Thesis Synchronization**:
   - `research_work/reports/phase6_noise_report.md` (dynamically generated report evaluating H3)
   - Thesis manuscript update in `Book_Writing/` (integrating noise findings into Chapter VII / Discussion)
   - `Overview/research_consistency_overview.md` upgraded to mark Phase 06 as `Completed & Verified`.

---

## 8. Step-by-Step Execution Sequence

* [x] **Step 1**: Implement `research_work/tests/test_noise_experiment.py` and verify all tests pass.
* [x] **Step 2**: Implement `research_work/src/noise_experiment.py` with fold-isolated feature selection, nested uniform noise injection, and two-stage aggregation.
* [x] **Step 3**: Execute `python research_work/src/noise_experiment.py` across all 10 folds, 6 models, 2 representations, 35 conditions, and 5 seeds.
* [x] **Step 4**: Validate generated outputs (20,520 raw rows, 420 summary rows, 300 DPI figures).
* [x] **Step 5**: Formally evaluate Hypothesis H3 (comparing primary grid vs. matched-burden control).
* [x] **Step 6**: Update thesis manuscript in `Book_Writing/` and `Overview/research_consistency_overview.md`.
