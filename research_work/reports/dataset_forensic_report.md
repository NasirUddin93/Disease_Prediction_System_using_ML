# E1: Dataset Forensic Audit & Integrity Report

**Investigation Date**: 2026-09-18  
**Dataset Reference**: Kaggle Disease Prediction Using Machine Learning (`kaushil268`)  
**Investigated Files**: 
- `Training.csv` (4920 rows, 134 cols)
- `Testing.csv` (42 rows, 133 cols)

---

## 1. Executive Summary & Core Forensic Findings

An exhaustive, code-verified empirical forensic analysis was executed on the downloaded dataset files to establish ground-truth characteristics prior to any model training. The audit confirms three critical findings:

1. **Discrepancy in Disease Classes (41 vs. 42)**:
   - The official Kaggle dataset page states that the dataset covers **42 diseases**.
   - Code verification confirms that both `Training.csv` and `Testing.csv` contain exactly **41 unique disease classes**. There is no 42nd disease class in either file.
2. **High Synthetic Redundancy (93.82% Duplication)**:
   - Across the 4,920 training records, there are **only 304 unique symptom vectors**.
   - Exactly **4616 out of 4,920 records (93.82%)** are verbatim duplicate copies of other rows. Each disease consists of only 5 to 10 unique symptom patterns replicated ~12 to 24 times.
3. **Severe Train-Test Data Leakage (97.62% Overlap)**:
   - Of the 42 records in `Testing.csv`, exactly **41 (97.62%) are exact duplicates of rows already present in `Training.csv`**.
   - Evaluating on the original Kaggle split merely measures a classifier's capacity to memorize 41 seen vectors, explaining why prior published studies trivially report 100% accuracy.
4. **Structural Artifact Column (`Unnamed: 133`)**:
   - `Training.csv` contains an empty 134th column caused by a trailing comma in the raw CSV header line. It contains 100% null values and must be dropped during preprocessing.

---

## 2. Forensic Metric Summary Table

| Forensic Dimension | Empirical Value | Scientific Significance |
| :--- | :--- | :--- |
| **Raw Training Records** | `4920` | Standard cohort size |
| **Raw Testing Records** | `42` | Extremely small test sample (~1 sample per disease) |
| **Clean Symptom Features** | `132` | Binary indicators ($x_i \in \{0, 1\}$) |
| **Artifact Columns Detected** | `['Unnamed: 133']` | Dropped (`Unnamed: 133`) |
| **Disease Prognosis Classes** | `41` | Multi-class target (Kaggle stated 42; reality is 41) |
| **Training Class Balance** | **Perfect (120 per class)** | Class balance ratio = 1.0 (no class imbalance in train) |
| **Unique Symptom Vectors in Train** | `304` | True distinct medical profiles |
| **Exact Duplicate Rows in Train** | `4616` (93.82%) | Extreme synthetic replication |
| **Contradictory Symptom Vectors** | `0` | Zero label contradictions (dataset is fully deterministic) |
| **Test Records Verbatim in Train** | `41` / `42` (97.62%) | **Severe data leakage** in original Kaggle split |
| **Symptom Count per Patient (Mean ± SD)** | `7.45 \pm 1.4` (Range: 3–17) | Sparse presentation (mean ~7.45 of 132 symptoms) |
| **Zero-Variance Features** | `1` | All 132 symptoms appear at least once |
| **High Collinearity Pairs (|r| ≥ 0.85)** | `95 pairs` | Noticeable symptom co-occurrence clusters |

---

## 3. Discrepancy & Anomaly Analysis

### 3.1 The 41 vs. 42 Disease Label Discrepancy
The Kaggle dataset description published by `kaushil268` specifies:
> *"The dataset consists of 42 diseases and 132 symptoms."*

Our forensic audit verifies that:
- `Training.csv`: Contains 4,920 rows and **41 unique diseases**, with each disease having exactly 120 samples ($41 \times 120 = 4,920$).
- `Testing.csv`: Contains 42 rows and **41 unique diseases**. Forty diseases have exactly 1 test sample, and exactly one disease (`Fungal infection`) has 2 test samples ($40 \times 1 + 1 \times 2 = 42$).
- **Conclusion**: The "42 diseases" claim in the Kaggle documentation is an error caused by confusing the **42 test set rows** with the number of classes.

### 3.2 Feature Header Whitespace Anomalies
Several feature columns in the CSV headers contain trailing or leading spaces:
- `'dischromic _patches'` (space before underscore)
- `'spotting_ urination'` (space after underscore)
- `'foul_smell_of urine'` (space after 'of')

These must be strictly stripped during ingestion to ensure consistent feature naming.

---

## 4. Class Distribution & Pattern Redundancy

Each disease class in the training dataset has exactly 120 samples, but only **5 to 10 unique symptom patterns**:

```
Sample Classes & True Pattern Diversity:
• Fungal infection:             5 unique patterns (Replicated 24.0x)
• Allergy:                      5 unique patterns (Replicated 24.0x)
• AIDS:                         5 unique patterns (Replicated 24.0x)
• Heart attack:                 5 unique patterns (Replicated 24.0x)
• GERD:                         7 unique patterns (Replicated 17.1x)
• Malaria:                      8 unique patterns (Replicated 15.0x)
• Diabetes:                     9 unique patterns (Replicated 13.3x)
• Dengue:                      10 unique patterns (Replicated 12.0x)
• Hepatitis D:                 10 unique patterns (Replicated 12.0x)
```

*(Full table exported to `class_distribution.csv`).*

---

## 5. Symptom Statistics & Clinical Presentations

- **Symptom Sparsity**: On average, a patient record presents only **7.45 out of 132 symptoms** (minimum 3, maximum 17).
- **Top 5 Most Frequent Symptoms**:
  - `fatigue`: 39.3% of patient records
  - `vomiting`: 38.9% of patient records
  - `high_fever`: 27.7% of patient records
  - `loss_of_appetite`: 23.4% of patient records
  - `nausea`: 23.3% of patient records
- **High Collinearity Examples**:
  - `nodal_skin_eruptions` & `dischromic _patches`: Pearson r = 0.8864
  - `shivering` & `watering_from_eyes`: Pearson r = 0.8864
  - `muscle_wasting` & `patches_in_throat`: Pearson r = 0.8864
  - `muscle_wasting` & `extra_marital_contacts`: Pearson r = 0.8864
  - `weight_gain` & `cold_hands_and_feets`: Pearson r = 0.9461

---

## 6. Methodological Implications for the Master's Thesis

1. **Invalidity of the Standard Kaggle Split**:
   Evaluating solely on `Testing.csv` is scientifically invalid for a Master's thesis because **41 of the 42 test cases are verbatim training duplicates**.
2. **Requirement for Three Evaluation Protocols**:
   - **Protocol A (Kaggle Split)**: Serves only as a reproducibility baseline to verify the literature's 100% accuracy claim.
   - **Protocol B (Standard Stratified CV)**: Demonstrates the effect of random partitioning when duplicates are present.
   - **Protocol C (Duplicate-Aware / Group-Aware CV)**: The primary benchmark, where identical symptom vectors are clustered into groups and assigned strictly to either training or validation folds, guaranteeing zero pattern leakage.
3. **Affirmation of Research Direction**:
   Because the clean unique dataset contains only 304 patterns across 41 classes, real-world clinical deployment requires models that do not fail when symptoms are **sparse (cold-start)**, **noisy (misreported)**, or **ambiguous (overlapping)**. This validates our proposed experiments on **Sparsity Decay (E5)**, **Noise Degradation (E6)**, and **Probability Calibration (E7)**.
