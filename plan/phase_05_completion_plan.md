# Phase 05 Completion Plan: Symptom Sparsity & Cold-Start Robustness

**Project Title**: *Disease Prediction System Using Machine Learning: A Robust, Explainable, and Uncertainty-Aware Framework Under Real-World Symptom Sparsity and Noise*  
**Author**: Nasir uddin (Registration No: `23880300014`, Session: `2023-2024`)  
**Supervisor**: Husne Ara Rubaiyeat, Assistant Professor  
**Department**: Department of Computer Science, National University, Gazipur, Bangladesh  
**Phase Target**: Phase 05 — Clinical Symptom Sparsity & Cold-Start Robustness Engine  
**Status**: Formal Execution Blueprint & Methodological Audit Plan  

---

## 1. Executive Summary & Objective

In real-world Clinical Decision Support Systems (CDSS), such as emergency triage, patient intake portals, and telemedicine screening, patients rarely present an exhaustive checklist of symptoms. Instead, they present an acute, restricted cluster of primary chief complaints.

While Phase 4 established that a model requires a static budget of $k = 50$ symptoms to maintain $\ge 95\%$ Macro-F1 retention, **Phase 05 measures runtime dynamic degradation**:
> *How does diagnostic performance (Top-1, Top-3, Top-5, Macro-F1, Brier score) degrade when an intake patient presents only $s \in [1, 10]$ active symptoms out of their true pathology?*

### Critical Scientific Scope Distinction
This experiment measures algorithmic sensitivity under **controlled synthetic masking** of an existing benchmark dataset under Protocol C (duplicate-aware cross-validation). It characterizes algorithmic robustness under information loss at inference time; it does not claim to validate performance on real prospective human patient cohorts.

---

## 2. Global Experimental Architecture

```
┌─────────────────────────────────┐      ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│     1. Methodological Audit     │ ───► │  2. Automated Test Validation   │ ───► │   3. Full Reproducible Run      │
│  - Fold-isolated feature select │      │  - Nested subset masking test   │      │  - 10 folds × 2 reps × 6 models │
│  - Nested permutation masking   │      │  - Group-split leakage check    │      │  - 8 budgets × 5 seeds          │
│  - Two-stage seed/fold summary  │      │  - Metric bounds & class checks │      │  - Exactly 4,920 raw rows       │
└─────────────────────────────────┘      └─────────────────────────────────┘      └─────────────────────────────────┘
                                                                                                │
                                                                                                ▼
┌─────────────────────────────────┐      ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│ 6. Consistency Status Upgrade   │ ◄─── │ 5. Thesis Chapter VI Update     │ ◄─── │  4. Artifact & Figure Freeze    │
│  - Mark Phase 05 as Completed   │      │  - Replace provisional tables   │      │  - 300 DPI figures to Book/     │
│  - Update Master README.md      │      │  - Reconcile actual Top-k claims│      │  - Dynamic markdown report      │
└─────────────────────────────────┘      └─────────────────────────────────┘      └─────────────────────────────────┘
```

---

## 3. Methodological Audit & Core Fixes (P0 & P1)

### Priority P0: Critical Scientific Corrections

#### 1. Eliminate Feature-Selection Leakage for the 50-Feature Space
* **Identified Vulnerability**: The original script imported `selected_features/mutual_information.csv`, which was globally generated using all records in Phase 4. Evaluating on validation folds using features selected across the entire dataset introduces subtle information leakage.
* **Methodological Fix**:
  * In each of the 10 Protocol C folds, compute feature rankings (Mutual Information) strictly using $X_{\text{train}}$ and $y_{\text{train}}$ of that fold.
  * Extract the fold-specific top-50 feature subset.
  * Transform both $X_{\text{train}}$ and $X_{\text{val}}$ using these fold-specific 50 features.
  * Train and evaluate models on this leakage-free representation.
  * Maintain the **132-feature representation** as the primary clinical baseline, using the fold-isolated 50-feature space for leakage-free dimensionality comparison.

#### 2. Clarification of Baseline ($k$ vs. $s$) & Terminology
* Rename the baseline condition from `Full` to `Natural_Unmasked`.
* Clearly differentiate the two distinct quantities:
  * **$k$**: The total feature dimensions available to the model ($k = 132$ or $k = 50$).
  * **$s$**: The maximum active symptom budget cap at inference ($s \in \{10, 8, 6, 5, 4, 3, 2, 1\}$).
* The unmasked cohort natural mean is $7.45 \pm 3.59$ active symptoms. Therefore, $s$ is a **maximum cap**:
  * If a record presents $P \le s$ symptoms, all naturally present symptoms are preserved.
  * If a record presents $P > s$ symptoms, exactly $s$ symptoms are retained.

#### 3. Dynamic Reconciliation of Scientific Claims
* Eliminate sweeping universal claims (e.g., claiming all models maintain $>85\%$ Top-3 at $s=3$, whereas SVM achieves $62.38\%$ and XGBoost achieves $67.90\%$).
* All findings in `phase5_sparsity_report.md` and Chapter VI must be dynamically calculated and populated directly from the finalized experimental output CSV.

---

### Priority P1: Statistical & Engineering Rigour

#### 4. Two-Stage Hierarchical Aggregation (Folds vs. Seeds)
* **Identified Vulnerability**: Previously, summary rows conflated independent cross-validation fold variance with stochastic perturbation seed variance.
* **Methodological Fix**:
  1. **Stage 1 (Seed Averaging)**: For each fold $f \in \{1..10\}$, model $m$, representation $r$, and budget $s$, calculate the mean performance across the 5 stochastic masking seeds:
     $$\bar{m}_{f, s} = \frac{1}{5} \sum_{\text{seed}=1}^5 m_{f, s, \text{seed}}$$
  2. **Stage 2 (Fold Cross-Validation Statistics)**: Calculate the fold-level mean and sample standard deviation across the 10 folds:
     $$\mu_s = \frac{1}{10} \sum_{f=1}^{10} \bar{m}_{f, s}, \quad \sigma_s = \sqrt{\frac{1}{9} \sum_{f=1}^{10} (\bar{m}_{f, s} - \mu_s)^2}$$
  3. Separately record the perturbation variance across seeds to report sensitivity to random symptom selection.

#### 5. Nested Subsetting for Masking Fairness & Continuity
* **Identified Vulnerability**: Selecting symptoms independently at $s=3$ and $s=5$ may pick unrelated disjoint symptoms for the same seed, breaking degradation monotonicity.
* **Methodological Fix**: For each validation sample and seed, generate a single reproducible random permutation of its active symptom indices. For any budget $s$, select the prefix of length $\min(s, \text{len}(\text{active}))$.
  * **Guarantee**: Symptoms preserved at $s=3$ are a strict mathematical subset of symptoms preserved at $s=5$ for that seed ($\mathcal{S}_{s=3} \subset \mathcal{S}_{s=5}$).

#### 6. Class Alignment & Metric Precision
* Enforce global 41-class ordering across probability outputs using `clf.classes_`.
* Compute macro-averaged F1 score over the explicit 41-class label set with `zero_division=0`.
* Treat multi-class Brier score as a proper probability scoring rule:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^{41} (p_{ic} - y_{ic})^2$$
  (Documented as probability error loss, not standalone calibration proof).

#### 7. Centralized Output Paths
* Direct figure exports simultaneously to:
  * `research_work/reports/figures/`
  * `Book_Writing/figures/` (eliminating references to obsolete paths).

---

## 4. Experimental Configuration & Row Count Audit

### Frozen Hyperparameters
* **Dataset**: `Training.csv` (4,920 records, 41 balanced classes, 304 unique symptom vectors).
* **Validation**: Protocol C (Duplicate-Aware Group 10-Fold CV; identical symptom vectors isolated in single folds).
* **Models (6)**:
  1. `Logistic Regression`: $L_2$ regularization, $C=1.0$, `max_iter=300`.
  2. `Random Forest`: 100 trees, `random_state=42`.
  3. `SVM (RBF)`: Radial basis function kernel, probability estimates enabled.
  4. `MLP`: Hidden layers (128, 64), early stopping, `random_state=42`.
  5. `XGBoost`: 60 estimators, `max_depth=4`, `random_state=42`.
  6. `Multinomial Naive Bayes`: Laplace smoothing $\alpha=1.0$.
* **Sparsity Budgets ($s$)**: `Natural_Unmasked`, 10, 8, 6, 5, 4, 3, 2, 1.
* **Perturbation Seeds (5)**: `42`, `101`, `2024`, `777`, `999`.
* **Representations (2)**: Full 132 features, Fold-specific 50 features.

### Mathematical Verification of Raw Output Rows
$$N_{\text{unmasked}} = 10 \text{ folds} \times 2 \text{ representations} \times 6 \text{ models} \times 1 \text{ baseline} = 120 \text{ rows}$$
$$N_{\text{masked}} = 10 \text{ folds} \times 2 \text{ representations} \times 6 \text{ models} \times 8 \text{ budgets} \times 5 \text{ seeds} = 4{,}800 \text{ rows}$$
$$N_{\text{total}} = 120 + 4{,}800 = \mathbf{4{,}920 \text{ rows}}$$

### Summary Aggregation Rows
$$N_{\text{summary}} = 2 \text{ representations} \times 6 \text{ models} \times 9 \text{ budgets} = \mathbf{108 \text{ rows}}$$

---

## 5. Automated Verification Test Suite

Before executing the full benchmark, build and execute `research_work/tests/test_sparsity.py` covering:

1. **Test Masking Preservation**: For records with $\le s$ active symptoms, output identical to input.
2. **Test Masking Cap**: For records with $> s$ active symptoms, exactly $s$ positive indicators remain.
3. **Test Nested Subset Invariant**: For sample $i$ and seed $\sigma$, active indices at $s=3$ must be a subset of active indices at $s=5$.
4. **Test Immutability**: Input matrices are not modified in-place; outputs remain strictly binary $\{0, 1\}$.
5. **Test Zero Leakage**: Intersection of unique symptom pattern hashes between training and validation folds is $\emptyset$.
6. **Test Ranking Monotonicity**: For all predictions, $\text{Top-5 Accuracy} \ge \text{Top-3 Accuracy} \ge \text{Top-1 Accuracy}$.
7. **Test Probabilistic Validity**: Probabilities sum to $1.0 \pm 10^{-5}$ across 41 classes with no NaN/Inf values.

---

## 6. Step-by-Step Implementation Sequence

### Step 1: Code Refactoring (`research_work/src/sparsity_experiment.py`)
- Implement `apply_nested_symptom_sparsity()` ensuring nested prefix selection.
- Implement fold-isolated mutual information feature ranking on $X_{\text{train}}$.
- Align probability matrix dimensions with global 41-class label mapping.
- Implement two-stage hierarchical aggregation (seed average $\to$ fold mean $\pm$ fold SD).
- Update figure generators to save directly to `Book_Writing/figures/`.

### Step 2: Implement Test Suite (`research_work/tests/test_sparsity.py`)
- Implement comprehensive unit tests for masking, grouping, metrics, and nested subset properties.
- Execute via `python -m unittest research_work/tests/test_sparsity.py`.

### Step 3: Execute Finalized Experiment Run
- Execute the full benchmark:
  ```bash
  python research_work/src/sparsity_experiment.py
  ```
- Verify execution produces:
  - `research_work/reports/sparsity_fold_results.csv` (exactly 4,920 rows).
  - `research_work/reports/sparsity_summary.csv` (exactly 108 rows).
  - `research_work/reports/phase5_sparsity_report.md`.
  - `research_work/reports/figures/sparsity_decay_curves.png` & `Book_Writing/figures/sparsity_decay_curves.png`.
  - `research_work/reports/figures/sparsity_rep_comparison.png` & `Book_Writing/figures/sparsity_rep_comparison.png`.

### Step 4: Validate Data Integrity & Reconcile Metrics
- Assert completeness of all 4,920 rows with zero missing values.
- Verify metrics fall in expected theoretical bounds ($0 \le \text{Acc, F1} \le 1$, $\text{BS} \ge 0$).
- Inspect two-stage standard deviation values.

### Step 5: Update Master Thesis Manuscript (`Book_Writing/Chap6/Chapter6.tex`)
- Replace provisional numerical values in **Table 6.1** with finalized fold-level mean $\pm$ SD.
- Remove all provisional caveats (*"Curves reflect provisional outputs..."*).
- Update analytical text in Sections 6.2 and 6.3 with verified numerical observations.

### Step 6: Update Repository Overview & Commit
- Update `Overview/research_consistency_overview.md` to change Phase 05 status from `Reconciled (Provisional)` to `Completed (Verified)`.
- Update repository root `README.md`.
- Commit all finalized artifacts to git.

---

## 7. Phase 05 Completion Checklist

- [x] **1. Methodological Audit Complete**: Fold-level MI selection, nested masking, two-stage aggregation coded.
- [x] **2. Automated Test Suite Passes**: All assertions in `test_sparsity.py` succeed with zero errors.
- [x] **3. Reproducible Run Executed**: Script executes cleanly across all 10 folds and 5 seeds.
- [x] **4. Raw Results Verified**: `sparsity_fold_results.csv` verified to contain exactly 4,920 rows.
- [x] **5. Summary Metrics Generated**: `sparsity_summary.csv` verified to contain exactly 108 rows.
- [x] **6. Publication Figures Synchronized**: 300 DPI figures updated in both `research_work/reports/figures/` and `Book_Writing/figures/`.
- [x] **7. Dynamic Markdown Report Generated**: `phase5_sparsity_report.md` updated with exact verified figures.
- [x] **8. Chapter VI Reconciled**: Thesis text and Table 6.1 updated, provisional notes removed.
- [x] **9. Consistency Overview Upgraded**: Phase 05 officially marked as completed and verified.
