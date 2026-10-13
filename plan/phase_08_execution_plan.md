# Phase 08 Execution Plan: Explainable AI (XAI) & Clinical Literature Alignment (Hypothesis H5)

**Document Status**: Official Methodological Blueprint  
**Author**: Nasir uddin (Registration No: `23880300014`, Session: `2023-2024`)  
**Supervisor**: Husne Ara Rubaiyeat, Assistant Professor, Department of Computer Science, National University, Bangladesh  
**Target Output**: Phase 08 Empirical Suite (`research_work/src/shap_experiment.py`, `research_work/reports/phase8_shap_report.md`), Thesis Manuscript Chapter VII  

---

## 1. Executive Research Objective & Hypothesis Formulation

In healthcare decision support systems (CDSS), high predictive accuracy and calibrated probabilities are necessary but insufficient for clinical adoption. Machine learning models must be transparent, interpretable, and aligned with pathophysiological reasoning. Black-box predictions without attributable symptom evidence risk clinical mistrust and unspotted failure modes.

### Research Hypothesis H5 Postulate:
> *Model decisions explained via additive feature attributions (SHAP) align with recognized primary diagnostic criteria from authoritative medical literature (WHO and CDC guidelines) for representative high-prevalence diseases, and ensemble architectures exhibit significantly higher explanation stability across cross-validation folds than single-tree or unregularized linear baselines.*

---

## 2. Experimental Design & Methodology

### 2.1 Benchmark Dataset & Zero-Leakage Protocol C
* **Dataset**: `Training.csv` (4,920 records, 41 balanced classes, 304 unique symptom templates).
* **Cross-Validation**: Protocol C Duplicate-Aware Group 10-Fold Cross-Validation, ensuring outer test folds are evaluated on unseen symptom templates.
* **Feature Spaces**: Full 132-symptom feature space and fold-isolated 50-symptom space ($k=50$).

### 2.2 Model Architecture Coverage
Explainability will be audited across diverse model families:
1. **Tree Ensembles (TreeSHAP)**: Random Forest, Extra Trees, XGBoost.
2. **Single-Tree Baselines (TreeSHAP)**: Decision Tree.
3. **Linear Classifiers (LinearSHAP)**: Logistic Regression (L2-regularized).
4. **Neural & Kernel Models (KernelSHAP / DeepSHAP)**: Multi-Layer Perceptron (MLP).

### 2.3 Core Investigation Streams

#### Stream 1: Global Symptom Attribution & Disease Profiling
* Compute mean absolute SHAP values across the entire cohort:
  $$\bar{\phi}_j = \frac{1}{N} \sum_{i=1}^N |\phi_{i,j}^{(y_i)}|$$
* Extract the top-10 global discriminators across all 41 classes.
* Generate class-specific SHAP summary beeswarm and bar plots for each disease class.

#### Stream 2: Local Patient-Level Diagnostic Attribution
* Deconstruct individual patient predictions using SHAP Waterfall plots and Force plots:
  $$f(\mathbf{x}) = \phi_0 + \sum_{j=1}^M \phi_j(\mathbf{x})$$
* Quantify supporting symptom evidence ($\phi_j > 0$) vs. contraindicating or absent symptom evidence ($\phi_j < 0$).
* Evaluate cold-start cases under sparse presentation ($s = 3$ and $s = 5$ presenting symptoms).

#### Stream 3: Explanation Stability Under Resampling & Noise
* **Cross-Fold Stability**: Evaluate Jaccard similarity and Spearman rank correlation of top-5 attributed symptoms across all 10 Protocol C outer folds:
  $$J(\mathcal{S}_a, \mathcal{S}_b) = \frac{|\mathcal{S}_a \cap \mathcal{S}_b|}{|\mathcal{S}_a \cup \mathcal{S}_b|}$$
* **Perturbation Robustness**: Measure attribution drift when patients experience 10% false-negative dropout or 2% false-positive misreporting.

#### Stream 4: Clinical Concordance Audit (WHO / CDC Benchmark)
Benchmarking top-5 model-derived symptoms against clinical diagnostic guidelines for 6 representative reference diseases:
1. **Dengue**: WHO Dengue Guidelines (High fever, severe headache, retro-orbital pain, joint/muscle pain, rash).
2. **Malaria**: CDC Malaria Diagnosis Protocol (Chills, high fever, sweating, headache, nausea/vomiting).
3. **Typhoid**: WHO Enteric Fever Guide (Prolonged fever, abdominal pain, headache, constipation/diarrhea, fatigue).
4. **Bronchial Asthma**: GINA Guidelines (Wheezing, shortness of breath, cough, chest tightness).
5. **Diabetes**: ADA Clinical Criteria (Polyuria, polydipsia, weight loss, fatigue, blurred vision).
6. **Gastroesophageal Reflux Disease (GERD)**: ACG Clinical Guidelines (Heartburn, acid regurgitation, chest pain, dysphagia).

* **Metrics**:
  * Clinical Concordance Ratio: $\frac{|\text{Model Top-5} \cap \text{Guideline Criteria}|}{5}$
  * Detection of dataset-specific synthetic artifacts (spurious correlations resulting from rule-based template generation).

---

## 3. Implementation Plan & Deliverables

### 3.1 Script Deliverable
* File: `research_work/src/shap_experiment.py`
* Capabilities:
  - Fits models on Protocol C folds.
  - Computes exact TreeSHAP and LinearSHAP attributions.
  - Generates cross-fold stability matrices (Jaccard and Spearman).
  - Performs automated clinical concordance scoring against a structured WHO/CDC reference dictionary.
  - Outputs summary CSV tables and high-resolution figures.

### 3.2 Output Artifacts
1. `research_work/reports/shap_global_importance.csv`: Mean absolute SHAP values per feature per disease.
2. `research_work/reports/shap_fold_stability.csv`: Fold-to-fold Jaccard and rank correlation metrics.
3. `research_work/reports/clinical_concordance_audit.csv`: Model top-5 vs. clinical guidelines concordance rates.
4. `research_work/reports/figures/shap_summary_top_diseases.png`: Multi-panel global beeswarm / bar plot.
5. `research_work/reports/figures/shap_stability_comparison.png`: Stability comparison across model architectures.
6. `research_work/reports/phase8_shap_report.md`: Dynamic comprehensive Markdown report.

---

## 4. Verification Checklist & Success Criteria
- [ ] Protocol C zero-leakage cross-validation strictly maintained.
- [ ] TreeSHAP values satisfy additivity property ($\sum \phi_j = f(x) - \mathbb{E}[f(x)]$).
- [ ] Cross-fold explanation stability quantified for Random Forest, Extra Trees, Decision Tree, Logistic Regression, and MLP.
- [ ] Clinical concordance audit executed and documented against WHO/CDC references.
- [ ] Synthetic artifact symptoms identified and documented as dataset limitations.
- [ ] Results integrated into Chapter VII of the thesis manuscript.
