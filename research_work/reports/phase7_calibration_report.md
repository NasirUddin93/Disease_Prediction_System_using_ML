# Phase 7: Probability Calibration & Uncertainty Quantification Report

**Investigation Date**: October 2026  
**Protocol**: Protocol C (Duplicate-Aware Group 10-Fold CV, Zero Inner-to-Outer Leakage)  
**Sample Space**: 4,920 records, 41 balanced classes, 304 unique symptom templates  
**Total Evaluated Rows**: 2700 fold runs, aggregated into 270 summary conditions  

---

## 1. Executive Summary & Evaluation of Hypothesis H4

In clinical triage, accurate disease ranking must be paired with trustworthy probability estimates. Hypothesis H4 posits that *Platt scaling (sigmoid calibration) and Isotonic regression will reduce ECE and Brier score without degrading classification or candidate ranking, particularly for tree ensembles and neural networks.*

### Empirical Verdict on Hypothesis H4: **Partially Supported (Architecture-Dependent)**

Post-hoc calibration demonstrated clear efficacy on a subset of confirmatory models (Random Forest, MLP Neural Net, Extra Trees), while other models showed mixed or baseline-calibrated metrics.

Key empirical findings derived directly from the complete 10-fold cross-validation results:
- **Random Forest**: Native ECE = 0.1114. Best calibrated ECE delta = -0.0538 (Sigmoid: -0.0143, Isotonic: -0.0538). Best Brier delta = -0.0426.
- **XGBoost**: Native ECE = 0.0324. Best calibrated ECE delta = +0.0554 (Sigmoid: +0.1131, Isotonic: +0.0554). Best Brier delta = -0.0383.
- **MLP Neural Net**: Native ECE = 0.2680. Best calibrated ECE delta = -0.2164 (Sigmoid: -0.1812, Isotonic: -0.2164). Best Brier delta = -0.1152.
- **Decision Tree**: Native ECE = 0.1743. Best calibrated ECE delta = +0.1478 (Sigmoid: +0.1478, Isotonic: +0.1561). Best Brier delta = +0.1214.
- **Extra Trees**: Native ECE = 0.1042. Best calibrated ECE delta = -0.0412 (Sigmoid: +0.0031, Isotonic: -0.0412). Best Brier delta = -0.0652.

---

## 2. Quantitative Performance Table (Clean Baseline, Full 132 Space)

| Model Architecture | Calibration Mode | ECE-15 (Mean +- SD) | Multiclass Brier Score | Multiclass Log Loss | Top-1 Acc | Top-3 Acc | Top-5 Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | Isotonic | $0.3304 \pm 0.0963$ | $0.4699$ | $2.3931$ | $0.7479$ | $0.8648$ | $0.8855$ |
| **Decision Tree** | Native | $0.1743 \pm 0.0357$ | $0.3486$ | $6.0193$ | $0.8257$ | $0.8403$ | $0.8428$ |
| **Decision Tree** | Sigmoid | $0.3221 \pm 0.0744$ | $0.4926$ | $1.5906$ | $0.7040$ | $0.8184$ | $0.8367$ |
| **Dummy Control** | Isotonic | $0.0272 \pm 0.0001$ | $0.9934$ | $4.2453$ | $0.0000$ | $0.0000$ | $0.0000$ |
| **Dummy Control** | Native | $0.0271 \pm 0.0000$ | $0.9934$ | $4.2451$ | $0.0000$ | $0.0000$ | $0.0000$ |
| **Dummy Control** | Sigmoid | $0.0272 \pm 0.0001$ | $0.9932$ | $4.2342$ | $0.0000$ | $0.0000$ | $0.0000$ |
| **Extra Trees** | Isotonic | $0.0629 \pm 0.0283$ | $0.0186$ | $0.0714$ | $1.0000$ | $1.0000$ | $1.0000$ |
| **Extra Trees** | Native | $0.1042 \pm 0.0327$ | $0.0838$ | $0.1967$ | $0.9720$ | $0.9988$ | $1.0000$ |
| **Extra Trees** | Sigmoid | $0.1072 \pm 0.0293$ | $0.0250$ | $0.1221$ | $1.0000$ | $1.0000$ | $1.0000$ |
| **Logistic Regression** | Isotonic | $0.0639 \pm 0.0233$ | $0.0153$ | $0.0700$ | $1.0000$ | $1.0000$ | $1.0000$ |
| **Logistic Regression** | Native | $0.0282 \pm 0.0050$ | $0.0022$ | $0.0293$ | $1.0000$ | $1.0000$ | $1.0000$ |
| **Logistic Regression** | Sigmoid | $0.1030 \pm 0.0200$ | $0.0180$ | $0.1122$ | $1.0000$ | $1.0000$ | $1.0000$ |
| **MLP Neural Net** | Isotonic | $0.0517 \pm 0.0275$ | $0.0157$ | $0.0577$ | $1.0000$ | $1.0000$ | $1.0000$ |
| **MLP Neural Net** | Native | $0.2680 \pm 0.0752$ | $0.1294$ | $0.3792$ | $1.0000$ | $1.0000$ | $1.0000$ |
| **MLP Neural Net** | Sigmoid | $0.0869 \pm 0.0261$ | $0.0143$ | $0.0951$ | $1.0000$ | $1.0000$ | $1.0000$ |
| **Naive Bayes** | Isotonic | $0.0409 \pm 0.0278$ | $0.0131$ | $0.0458$ | $1.0000$ | $1.0000$ | $1.0000$ |
| **Naive Bayes** | Native | $0.0084 \pm 0.0118$ | $0.0026$ | $0.0093$ | $1.0000$ | $1.0000$ | $1.0000$ |
| **Naive Bayes** | Sigmoid | $0.0856 \pm 0.0259$ | $0.0134$ | $0.0932$ | $1.0000$ | $1.0000$ | $1.0000$ |
| **Random Forest** | Isotonic | $0.0575 \pm 0.0278$ | $0.0155$ | $0.0647$ | $1.0000$ | $1.0000$ | $1.0000$ |
| **Random Forest** | Native | $0.1114 \pm 0.0218$ | $0.0581$ | $0.1524$ | $0.9939$ | $1.0000$ | $1.0000$ |
| **Random Forest** | Sigmoid | $0.0971 \pm 0.0270$ | $0.0196$ | $0.1088$ | $0.9988$ | $1.0000$ | $1.0000$ |
| **SVM RBF** | Isotonic | $0.0918 \pm 0.0303$ | $0.0343$ | $0.1034$ | $1.0000$ | $1.0000$ | $1.0000$ |
| **SVM RBF** | Native | $0.3148 \pm 0.0161$ | $0.1110$ | $0.3873$ | $1.0000$ | $1.0000$ | $1.0000$ |
| **SVM RBF** | Sigmoid | $0.3993 \pm 0.0642$ | $0.3964$ | $0.7891$ | $0.8319$ | $0.9866$ | $1.0000$ |
| **XGBoost** | Isotonic | $0.0878 \pm 0.0319$ | $0.0906$ | $0.2986$ | $0.9477$ | $0.9684$ | $0.9842$ |
| **XGBoost** | Native | $0.0324 \pm 0.0094$ | $0.1289$ | $0.4509$ | $0.8831$ | $0.8990$ | $0.9148$ |
| **XGBoost** | Sigmoid | $0.1455 \pm 0.0352$ | $0.1098$ | $0.4107$ | $0.9441$ | $0.9526$ | $0.9538$ |

---

## 3. Paired Statistical Analysis on Confirmatory Models (H4 Testing)

| Model | Native ECE | Sigmoid ECE (Delta) | Isotonic ECE (Delta) | Native BS | Sigmoid BS (Delta) | Isotonic BS (Delta) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.1114 | 0.0971 (-0.0143) | 0.0575 (-0.0538) | 0.0581 | 0.0196 (-0.0385) | 0.0155 (-0.0426) |
| **XGBoost** | 0.0324 | 0.1455 (+0.1131) | 0.0878 (+0.0554) | 0.1289 | 0.1098 (-0.0191) | 0.0906 (-0.0383) |
| **MLP Neural Net** | 0.2680 | 0.0869 (-0.1812) | 0.0517 (-0.2164) | 0.1294 | 0.0143 (-0.1152) | 0.0157 (-0.1138) |
| **Decision Tree** | 0.1743 | 0.3221 (+0.1478) | 0.3304 (+0.1561) | 0.3486 | 0.4926 (+0.1441) | 0.4699 (+0.1214) |
| **Extra Trees** | 0.1042 | 0.1072 (+0.0031) | 0.0629 (-0.0412) | 0.0838 | 0.0250 (-0.0588) | 0.0186 (-0.0652) |

---

## 4. Robustness Transfer Under Distribution Shift (Sparsity & Noise)

- **Clean Calibration Transfer**: Calibrators trained on clean training data retain superior probability calibration when transferred to moderate noise (`noise_drop20` and `noise_fp2`).
- **Severe Sparsity Stress ($s=3$)**: Under extreme symptom sparsity, all models experience probability miscalibration (ECE rises), but sigmoid-calibrated models degrade far more gracefully than uncalibrated models.
- **Differential Diagnostic Buffer**: Top-3 and Top-5 candidate retrieval retains diagnostic coverage across all calibrated variants.

---

## 5. Verification & Reproducibility Metadata
- [x] Protocol C duplicate-aware grouping strictly maintained across all 10 outer folds.
- [x] Nested inner 5-fold StratifiedGroupKFold on outer train guarantees zero calibration leakage.
- [x] Evaluated across 9 model architectures and 2 feature spaces.
- [x] 15-bin fixed and 10-bin quantile ECE sensitivity checks verified.
- [x] Full 41-class out-of-fold probability vectors saved to calibration_oof_probabilities.csv.
