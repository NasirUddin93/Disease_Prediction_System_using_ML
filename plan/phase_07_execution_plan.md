# Phase 07 Execution Plan: Probability Calibration & Uncertainty Quantification (Hypothesis H4)

**Document Status**: Proposal & Methodological Blueprint  
**Author**: Nasir uddin (Registration No: `23880300014`, Session: `2023-2024`)  
**Supervisor**: Husne Ara Rubaiyeat, Assistant Professor, Department of Computer Science, National University, Bangladesh  
**Target Chapter**: `Book_Writing/Chap7/Chapter7.tex` / Research Phase E7  

---

## 1. Executive Research Objective & Hypothesis Formulations

In clinical decision support systems (CDSS), an accurate ranking of diseases must be accompanied by trustworthy, calibrated posterior probabilities $P(y = c \mid \mathbf{x})$. While standard cross-validation benchmarks measure *discriminative ability* (can the model identify the correct disease?), they fail to evaluate *probabilistic reliability* (does a reported 90% confidence correspond to a 90% true positive empirical frequency?). 

### Research Hypothesis H4 Postulate:
> *Post-hoc probability calibration (Platt scaling and Isotonic regression) will significantly reduce Expected Calibration Error (ECE) and multiclass Brier score without degrading Macro-F1 or Top-k candidate diagnostic inclusion, particularly for non-probabilistic models such as decision tree ensembles (Random Forest, XGBoost, Decision Tree, Extra Trees) and uncalibrated neural networks (MLP).*

### Confirmatory vs. Exploratory Model Sets:
* **Confirmatory Core (Testing H4)**:
  1. `Random Forest` (Tree ensemble with bagging)
  2. `XGBoost` (Gradient boosted trees)
  3. `MLP` (Multi-layer perceptron neural network)
  4. `Decision Tree` (Single partitioned tree)
  5. `Extra Trees` (Extremely randomized trees)
* **Probabilistic / Linear Baselines (Comparators)**:
  6. `Logistic Regression` (Parametric soft-margin baseline, well-calibrated natively)
  7. `Naive Bayes` (Multinomial conditional independence; typically overconfident near 0 and 1)
  8. `SVM (RBF)` (Support Vector Classifier; evaluated natively and via uncalibrated decision-function scores calibrated externally)
* **Negative Control**:
  9. `Dummy Classifier` (Stratified prior baseline, verifying non-trivial calibration gain)

---

## 2. Experimental Scope & Computational Mathematics

### 2.1 Benchmark Dataset & Zero-Leakage Protocol C
* **Benchmark Input**: `Training.csv` (4,920 records, 41 balanced classes, 304 unique symptom templates).
* **Partitioning**: Protocol C Duplicate-Aware Group 10-Fold Cross-Validation. Duplicate vectors are grouped by SHA-256 hash ensuring zero vector overlap across outer folds.
* **Testing Set Isolation**: `Testing.csv` is strictly excluded due to the forensically audited 97.6% duplicate leakage.

### 2.2 Feature Representations
1. **Full 132-Symptom Space**: Reference uncompressed clinical feature space.
2. **Fold-Specific 50-Symptom Space**: Feature selection via Mutual Information executed strictly on $X_{\text{outer\_train}}$ inside each outer fold, guaranteeing zero validation leakage.

### 2.3 Nested, Group-Aware Calibration Framework (Zero Inner-to-Outer Leakage)

To avoid optimistic calibration bias, the calibration mappings must never be learned on outer validation folds or on outer training data without out-of-fold cross-validation:

```
Outer Protocol C Split (Fold k = 1..10)
├── Outer Training Set (90% of groups, ~4,428 samples)
│   ├── Fold-Isolated Feature Selection (k=50 subset computed on Outer Train only)
│   ├── Base Model Training: Model fitted on Outer Train
│   └── Inner Calibration Loop (5-Fold Group-Aware CV on Outer Train):
│       ├── Inner Train (4/5): Fit base classifier replica
│       ├── Inner Validation (1/5): Predict out-of-fold (OOF) raw probabilities / logits
│       └── Post-Processing:
│           ├── Fit Platt Scaling (Sigmoid) on OOF predictions vs. y_outer_train
│           └── Fit Isotonic Regression on OOF predictions vs. y_outer_train
└── Outer Validation Set (10% of groups, ~492 samples, completely untouched)
    ├── Predict Native Probabilities: P_native = Base_Model.predict_proba(X_outer_val)
    ├── Predict Sigmoid Calibrated:   P_sigmoid = Sigmoid_Calibrator.predict_proba(P_native)
    └── Predict Isotonic Calibrated:  P_isotonic = Isotonic_Calibrator.predict_proba(P_native)
```

### 2.4 Multi-Class Calibration Mechanisms
For a 41-class classification problem:
1. **Platt Scaling (Sigmoid)**: Fits a logistic regression model over the base model's uncalibrated probabilities/logits:
   $$\hat{P}(y = c \mid \mathbf{x}) = \frac{1}{1 + \exp(A_c f_c(\mathbf{x}) + B_c)}$$
   followed by softmax/L1 normalization across all 41 classes so that $\sum_{c=1}^{41} \hat{p}_{ic} = 1$.
2. **Isotonic Regression**: Fits a non-parametric, monotonic step function $m_c$ minimizing squared error:
   $$\hat{p}_{ic} = m_c(f_c(\mathbf{x})), \quad \text{subject to } m_c(u) \ge m_c(v) \text{ whenever } u \ge v$$
   followed by L1 normalization across all 41 classes.
3. **SVM Handling**:
   - Native baseline: Evaluated with standard Platt scaling inside scikit-learn (`probability=True`).
   - Clean External calibration: Fitted directly on raw decision scores ($f(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x}) + b$) via nested inner OOF to benchmark true external calibration against internal heuristic.

---

## 3. Mathematical Metric Definitions

### 3.1 Top-Label Expected Calibration Error (ECE)
Partitions the predicted top-class confidence $\hat{p}_i = \max_c \hat{P}(y = c \mid \mathbf{x}_i)$ into $B = 15$ equally spaced bins $I_b = (\frac{b-1}{B}, \frac{b}{B}]$:
$$\mathrm{ECE}_{15} = \sum_{b=1}^{B} \frac{|S_b|}{N} \left| \mathrm{acc}(S_b) - \mathrm{conf}(S_b) \right|$$
where:
$$\mathrm{acc}(S_b) = \frac{1}{|S_b|} \sum_{i \in S_b} \mathbf{1}(\hat{y}_i = y_i), \quad \mathrm{conf}(S_b) = \frac{1}{|S_b|} \sum_{i \in S_b} \hat{p}_i$$
* **Sensitivity Check**: In addition to $B=15$, compute adaptive equal-quantile binning ($B_{\text{quantile}} = 10$) to confirm binning invariance.

### 3.2 Multiclass Brier Score
Strictly proper scoring rule measuring overall mean squared error in probability space across all $C = 41$ classes:
$$\mathrm{BS} = \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \left( \hat{p}_{ic} - \mathbf{1}(y_i = c) \right)^2$$
* Range: $[0, 2]$. Optimal: $0.0$.

### 3.3 Multiclass Log Loss (Cross-Entropy)
$$\mathrm{LogLoss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \mathbf{1}(y_i = c) \ln(\max(\hat{p}_{ic}, \epsilon))$$
where $\epsilon = 10^{-15}$ to prevent infinite loss under extreme overconfidence.

### 3.4 Discriminative Integrity Checks
To verify that calibration does not distort diagnostic sorting, compute:
* **Top-1 Accuracy**
* **Macro-F1 Score**
* **Top-3 Accuracy**
* **Top-5 Accuracy**

---

## 4. Robustness Extension: Calibration Transfer Under Sparsity & Noise

Does calibration learned on clean, complete clinical vectors hold up when a patient presents with incomplete or noisy symptoms?
We evaluate the clean-trained calibrator across 5 validation test regimes:
1. `clean_baseline`: Clean, unmasked validation vectors.
2. `sparsity_s3`: Severe acute intake restriction ($s = 3$ symptoms).
3. `noise_drop20`: 20% random symptom dropout ($p_{\text{drop}} = 0.20$).
4. `noise_fp2`: 2% spurious symptom activation ($p_{\text{fp}} = 0.02$).
5. `noise_comb`: Combined perturbation ($p_{\text{drop}} = 0.10, p_{\text{fp}} = 0.02$).

**Leakage Invariant**: Calibrators are fitted strictly on clean inner training folds. No noisy or sparse validation profiles are ever exposed during calibration fitting.

---

## 5. Row Count & Dataset Specification

### Primary Benchmark Matrix (Clean Baseline):
* Outer Folds: 10
* Representations: 2 (Full 132, Fold-Specific 50)
* Models: 8 substantive (+ 1 dummy = 9 models)
* Calibration Modes: 3 (`native`, `sigmoid`, `isotonic`)
* **Primary Rows**: $10 \times 2 \times 9 \times 3 = \mathbf{540 \text{ rows}}$

### Robustness Transfer Matrix (5 Conditions):
* Conditions: 5 (`clean`, `s3`, `drop20`, `fp2`, `comb`)
* Folds: 10
* Representations: 2
* Models: 8 substantive
* Modes: 3
* **Robustness Rows**: $5 \times 10 \times 2 \times 8 \times 3 = \mathbf{2,400 \text{ rows}}$

### Total Saved Artifacts:
* `calibration_fold_results.csv`: Contains both primary and robustness runs ($\sim 2,700$ rows).
* `calibration_oof_probabilities.csv`: Out-of-fold predicted probability vectors for all 4,920 records (allowing exact reconstruction of curves).
* `calibration_summary.csv`: Aggregated hierarchical mean $\pm$ fold standard deviation across all conditions.

---

## 6. Pre-Execution Test Suite (`test_calibration_experiment.py`)

Implement automated unit tests before launching the full run asserting:
1. **Group Invariant**: No duplicate pattern hash appears in both inner training and inner calibration validation.
2. **Probability Normalization**: For all predictions, $\sum_{c=1}^{41} \hat{p}_{ic} = 1.0 \pm 10^{-6}$ and $\hat{p}_{ic} \ge 0.0$.
3. **Monotonicity of Isotonic Transformation**: Single-class probabilities maintain non-decreasing rank ordering prior to normalization.
4. **Ranking Invariance Check**: In cases of pure monotonic scaling, Top-1 and Top-k predictions are preserved or documented if tied.
5. **ECE Formula Invariant**: ECE on a perfectly calibrated synthetic dataset returns $0.0$; ECE on an inverted prediction returns $1.0$.
6. **Class Alignment Invariant**: Probability columns map 1-to-1 with canonical alphabetically sorted classes $[0 \dots 40]$.

---

## 7. Deliverables & Figures (300 DPI)

1. **Source Code**:
   - `research_work/src/calibration_experiment.py`
   - `research_work/tests/test_calibration_experiment.py`
2. **Numerical Datasets**:
   - `research_work/reports/calibration_fold_results.csv`
   - `research_work/reports/calibration_oof_probabilities.csv`
   - `research_work/reports/calibration_summary.csv`
3. **Publication-Grade Visualizations**:
   - `reliability_diagrams.png`: 15-bin reliability plots comparing Native vs. Sigmoid vs. Isotonic across models.
   - `calibration_metric_comparison.png`: Paired ECE and Brier score reductions across model families.
   - `calibration_noise_robustness.png`: ECE and Brier score trajectories under sparsity ($s=3$) and noise perturbations.
4. **Formal Report & Thesis Update**:
   - `research_work/reports/phase7_calibration_report.md`
   - `Book_Writing/Chap7/Chapter7.tex` (adding Section 7.3 on Probability Calibration & Uncertainty Quantification)
   - `Overview/research_consistency_overview.md` (upgrading Phase 7 to `✅ Completed & Verified`)
   - `Book_Writing_Overleaf.zip` re-packaged with new figures and updated manuscript.

---

## 8. Step-by-Step Execution Sequence

* [ ] **Step 1**: Implement `research_work/tests/test_calibration_experiment.py` and verify all tests pass.
* [ ] **Step 2**: Implement `research_work/src/calibration_experiment.py` with nested group-aware calibration, 15-bin ECE calculation, and multiclass Brier score.
* [ ] **Step 3**: Execute small single-fold smoke test (`--smoke-test`) to verify runtime, memory, and output shapes.
* [ ] **Step 4**: Execute full 10-fold benchmark across 8 models, 2 representations, 3 calibration modes, and 5 transfer conditions.
* [ ] **Step 5**: Generate 300 DPI publication figures and `phase7_calibration_report.md`.
* [ ] **Step 6**: Statistically evaluate Hypothesis H4 (paired t-test / Wilcoxon signed-rank test on ECE and Brier score).
* [ ] **Step 7**: Update `Book_Writing/Chap7/Chapter7.tex`, `Overview/research_consistency_overview.md`, and rebuild `Book_Writing_Overleaf.zip`.
