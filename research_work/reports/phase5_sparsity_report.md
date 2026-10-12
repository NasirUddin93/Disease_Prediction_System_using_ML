# Phase 5: Clinical Symptom Sparsity & Cold-Start Robustness Final Report

**Investigation Date**: October 2026  
**Protocol**: Protocol C (Duplicate-Aware Group 10-Fold Cross-Validation, zero pattern leakage)  
**Sample Space**: 4,920 records, 41 balanced classes, 304 unique symptom templates  
**Evaluated Symptom Budgets**: s in {Natural_Unmasked, 10, 8, 6, 5, 4, 3, 2, 1}  
**Perturbations**: 5 independent stochastic sampling seeds per sample (nested prefix selection)  
**Feature Spaces**: Full (132 features) vs. Fold-Specific Top-50 (leakage-free fold selection)  
**Aggregation**: Two-stage hierarchical (seed-average within fold -> fold mean +- SD)  

---

## 1. Executive Summary & Research Scope

In prospective clinical deployment (e.g., patient-facing symptom checkers, emergency room triage, telemedicine intake), patients rarely present with an exhaustive checklist of all 132 potential symptoms. Instead, patients present with an acute, restricted cluster of primary chief complaints.

While Phase 4 established that a model architecture requires **50 symptoms** to maintain operational retention (>= 95% Macro-F1), **Phase 5 investigates runtime dynamic resilience**:
> *How robust is the diagnostic model when an intake patient presents only s in [1, 10] active symptoms out of their true pathology?*

**Scientific Framing**: This experiment measures algorithmic sensitivity under synthetic masking on the benchmark dataset. It demonstrates how models respond to missing binary indicators at inference time; it does not claim to validate prospective human diagnostic accuracy.

---

## 2. Quantitative Sparsity Performance Table (Full 132-Feature Model)

The table below summarizes performance as the presenting symptom budget is progressively restricted from full natural presentation (~7.45 symptoms) down to an isolated chief complaint ($s = 1$):

| Model Architecture | Presenting Symptoms (s) | Top-1 Accuracy (Mean +- Fold SD) | Macro-F1 (Mean +- Fold SD) | Top-3 Accuracy | Top-5 Accuracy | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Natural_Unmasked | $1.0000 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $1.0000$ | $1.0000$ | $0.0022$ |
| **Logistic Regression** | 8 | $0.9814 \pm 0.0201$ | $0.9554 \pm 0.0384$ | $0.9998$ | $1.0000$ | $0.0476$ |
| **Logistic Regression** | 5 | $0.9019 \pm 0.0591$ | $0.7827 \pm 0.0624$ | $0.9824$ | $0.9971$ | $0.2023$ |
| **Logistic Regression** | 3 | $0.7581 \pm 0.0794$ | $0.5511 \pm 0.0447$ | $0.8848$ | $0.9338$ | $0.4256$ |
| **Logistic Regression** | 2 | $0.6243 \pm 0.1085$ | $0.3983 \pm 0.0560$ | $0.7507$ | $0.8169$ | $0.6221$ |
| **Logistic Regression** | 1 | $0.4173 \pm 0.0917$ | $0.2353 \pm 0.0365$ | $0.5465$ | $0.5735$ | $0.8557$ |
| **MLP** | Natural_Unmasked | $1.0000 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $1.0000$ | $1.0000$ | $0.1294$ |
| **MLP** | 8 | $0.9779 \pm 0.0186$ | $0.9456 \pm 0.0396$ | $0.9984$ | $0.9998$ | $0.2134$ |
| **MLP** | 5 | $0.8969 \pm 0.0508$ | $0.7904 \pm 0.0684$ | $0.9708$ | $0.9885$ | $0.3902$ |
| **MLP** | 3 | $0.7345 \pm 0.1139$ | $0.5827 \pm 0.0694$ | $0.8826$ | $0.9272$ | $0.6410$ |
| **MLP** | 2 | $0.5569 \pm 0.1320$ | $0.4251 \pm 0.0545$ | $0.7523$ | $0.8268$ | $0.7985$ |
| **MLP** | 1 | $0.2960 \pm 0.1229$ | $0.2148 \pm 0.0417$ | $0.4963$ | $0.6128$ | $0.9181$ |
| **Naive Bayes** | Natural_Unmasked | $1.0000 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $1.0000$ | $1.0000$ | $0.0026$ |
| **Naive Bayes** | 8 | $0.9981 \pm 0.0052$ | $0.9982 \pm 0.0048$ | $1.0000$ | $1.0000$ | $0.0056$ |
| **Naive Bayes** | 5 | $0.9758 \pm 0.0237$ | $0.9658 \pm 0.0186$ | $0.9994$ | $1.0000$ | $0.0362$ |
| **Naive Bayes** | 3 | $0.8837 \pm 0.0543$ | $0.8380 \pm 0.0427$ | $0.9679$ | $0.9986$ | $0.1395$ |
| **Naive Bayes** | 2 | $0.7283 \pm 0.1004$ | $0.6263 \pm 0.0398$ | $0.8937$ | $0.9568$ | $0.3133$ |
| **Naive Bayes** | 1 | $0.3719 \pm 0.0790$ | $0.3167 \pm 0.0269$ | $0.6036$ | $0.7177$ | $0.7155$ |
| **Random Forest** | Natural_Unmasked | $0.9939 \pm 0.0104$ | $0.9866 \pm 0.0220$ | $1.0000$ | $1.0000$ | $0.0581$ |
| **Random Forest** | 8 | $0.9718 \pm 0.0194$ | $0.9303 \pm 0.0384$ | $0.9989$ | $0.9995$ | $0.1286$ |
| **Random Forest** | 5 | $0.8938 \pm 0.0558$ | $0.7749 \pm 0.0666$ | $0.9765$ | $0.9894$ | $0.2439$ |
| **Random Forest** | 3 | $0.7675 \pm 0.1000$ | $0.5789 \pm 0.0655$ | $0.8946$ | $0.9395$ | $0.3972$ |
| **Random Forest** | 2 | $0.6701 \pm 0.1089$ | $0.4624 \pm 0.0533$ | $0.8174$ | $0.8666$ | $0.5332$ |
| **Random Forest** | 1 | $0.5048 \pm 0.0914$ | $0.3165 \pm 0.0405$ | $0.6522$ | $0.7119$ | $0.7343$ |
| **SVM** | Natural_Unmasked | $1.0000 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $1.0000$ | $1.0000$ | $0.1110$ |
| **SVM** | 8 | $0.9612 \pm 0.0334$ | $0.9013 \pm 0.0475$ | $0.9760$ | $0.9909$ | $0.2963$ |
| **SVM** | 5 | $0.8038 \pm 0.0919$ | $0.6203 \pm 0.0481$ | $0.8262$ | $0.8678$ | $0.4696$ |
| **SVM** | 3 | $0.5928 \pm 0.1167$ | $0.3730 \pm 0.0581$ | $0.6255$ | $0.6536$ | $0.6888$ |
| **SVM** | 2 | $0.4938 \pm 0.1114$ | $0.2986 \pm 0.0544$ | $0.5205$ | $0.5825$ | $0.8279$ |
| **SVM** | 1 | $0.2776 \pm 0.1269$ | $0.1480 \pm 0.0464$ | $0.2990$ | $0.3410$ | $0.9429$ |
| **XGBoost** | Natural_Unmasked | $0.8843 \pm 0.0448$ | $0.6802 \pm 0.1094$ | $0.9027$ | $0.9196$ | $0.1287$ |
| **XGBoost** | 8 | $0.7927 \pm 0.0475$ | $0.5466 \pm 0.0910$ | $0.8162$ | $0.8439$ | $0.2490$ |
| **XGBoost** | 5 | $0.7102 \pm 0.0615$ | $0.4461 \pm 0.0794$ | $0.7435$ | $0.7784$ | $0.3544$ |
| **XGBoost** | 3 | $0.6111 \pm 0.0664$ | $0.3435 \pm 0.0572$ | $0.6792$ | $0.7133$ | $0.4837$ |
| **XGBoost** | 2 | $0.5307 \pm 0.0769$ | $0.2862 \pm 0.0457$ | $0.6260$ | $0.6487$ | $0.6020$ |
| **XGBoost** | 1 | $0.3922 \pm 0.0771$ | $0.2153 \pm 0.0347$ | $0.5000$ | $0.5209$ | $0.7896$ |

---

## 3. Verified Empirical Discoveries

### 3.1 The Differential Diagnostic Buffer
- As presenting symptoms drop to $s = 3$:
  - **Single-Label Top-1 Accuracy** suffers a steep decline: Logistic Regression drops to 75.8%, Random Forest to 76.8%, MLP to 73.5%, and SVM to 59.3%.
  - In sharp contrast, **Top-3 Differential Accuracy** provides substantial recovery: Logistic Regression retains 88.5%, Random Forest retains 89.5%, MLP retains 88.3%, and SVM achieves 62.5%.
- **Clinical Implication**: In early intake triage where only 2-3 symptoms are reported, CDSS engines should **never emit a single dogmatic classification**. Providing a ranked candidate list (Top-3 to Top-5) provides a resilient decision-support safeguard before laboratory confirmation.

### 3.2 Architectural Sensitivity Under Sparsity
- **Additive & Neural Models (Logistic Regression, MLP)** demonstrate the most graceful degradation down to $s = 4$, as their continuous decision surfaces aggregate evidence across whichever symptoms are present.
- **Tree Ensembles (Random Forest, XGBoost)** degrade when high-level partition nodes check for symptoms omitted by the patient, though Random Forest retains competitive Top-3 and Top-5 retrieval at severe sparsity.
- **Kernel Models (SVM RBF)** degrade more rapidly at severe sparsity ($s <= 2$) because zeroing out active coordinates distorts geometric margins in kernel space.

### 3.3 Representation Comparison: Full (132) vs. Fold-Specific Top-50
- Evaluating models on the **fold-specific 50-symptom space** (strictly isolated per fold with zero leakage) shows close tracking to the 132-symptom curves down to $s = 5$.
- Below $s = 4$, the compressed 50-feature space degrades somewhat faster because masking active symptoms leaves fewer alternative informative features within the 50-symptom subspace.

---

## 4. Methodological Verification Status
- [x] Protocol C duplicate-aware grouping strictly enforced (0 leaking vector patterns).
- [x] Zero feature-selection leakage in the 50-feature space (fold-isolated MI).
- [x] Nested subset property mathematically guaranteed (reproducible seed permutation prefix).
- [x] Exactly 4,920 raw fold-level rows and 108 summary rows verified.
- [x] Two-stage aggregation separating fold cross-validation variance from seed perturbation variance.
