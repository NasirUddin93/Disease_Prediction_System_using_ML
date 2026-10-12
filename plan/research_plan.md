# Master's Research Plan: Disease Prediction System Using Machine Learning

**Official Topic**: *Disease Prediction System Using Machine Learning*  
**Research Subtitle**: *A Robust, Explainable and Uncertainty-Aware Machine Learning Framework for Symptom-Based Multi-Disease Prediction*  
**Degree Level**: Master of Science (M.Sc. / M.S. Thesis & Scholarly Peer-Reviewed Publication)  
**Target Dataset**: [Kaggle: Disease Prediction Using Machine Learning (kaushil268)](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)

---

## 1. Research Vision

### 1.1 Core Research Question
> **"Can machine learning reliably predict multiple diseases from reported symptoms when the symptom information is incomplete, noisy, and potentially ambiguous?"**

### 1.2 Conceptual Pipeline
Instead of a naive, direct black-box mapping:
$$\text{Symptoms} \longrightarrow \text{ML} \longrightarrow \text{Disease}$$

This research formulates a robust, uncertainty-aware, and explainable decision-support pipeline:

```
                  Patient Reported Symptoms
                             │
                             ▼
                     Data Preprocessing
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
      Complete Symptoms             Incomplete Symptoms
      (Full Profile)                (Sparse / Chief Complaints)
              │                             │
              └──────────────┬──────────────┘
                             ▼
                       ML Prediction
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
          Disease 1      Disease 2      Disease 3
         Probability    Probability    Probability
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    Uncertainty Analysis
              (Calibration: ECE, Brier Score)
                             │
                             ▼
                      Explainability
              (SHAP Feature Attribution &
               Literature Association Check)
                             │
                             ▼
              Clinical Decision Support Output
         (Differential Top-k, Evidence Summary,
               and Medical Safety Disclaimer)
```

> [!IMPORTANT]
> **Clinical Role**: The resulting framework is explicitly designed and evaluated as a **clinical decision-support and research prototype**, not as an autonomous medical diagnostic tool.

---

## 2. Research Problem Statement

The research problem is structured around five foundational clinical and methodological challenges:

1. **Problem 1 — Multi-Disease Classification Under Overlapping Symptomatology**:
   The dataset encompasses numerous disease entities that share non-specific, overlapping symptoms (e.g., fever, headache, fatigue, nausea). Symptoms rarely map uniquely to a single disease, demanding multi-class differential ranking rather than brittle single-label classification.

2. **Problem 2 — Dataset Quality, Duplication & Generalization Validity**:
   Published benchmarks and online implementations on this dataset commonly report near-perfect or 100% accuracy. The research investigates whether this performance reflects genuine generalization or the memorization of recurrent synthetic symptom patterns caused by underlying row duplication and train/test leakage.

3. **Problem 3 — Symptom Sparsity & Cold-Start Presentations**:
   In realistic clinical encounters, patients present 3 to 10 chief complaints, not an exhaustive inventory of 132 symptoms. The research evaluates how diagnostic performance degrades as the number of available symptoms is restricted.

4. **Problem 4 — Noisy & Erroneous Symptom Reporting**:
   Patient-reported symptoms are prone to recall bias, omission (false negatives), somatic over-reporting/hypochondria (false positives), and misunderstanding. The framework must quantify model resilience under controlled stochastic perturbations.

5. **Problem 5 — Interpretability & Uncertainty Deficit**:
   A deterministic output (e.g., *"Dengue — 92%"*) is unacceptable in clinical workflows. Decision support requires calibrated probabilities (Top-1, Top-3, Top-5 differential sets), transparent supporting/opposing symptom attributions, and clear confidence indicators.

---

## 3. Main & Specific Research Objectives

### Primary Objective
To develop and systematically evaluate a robust, interpretable, and uncertainty-aware machine-learning framework for multi-disease prediction from patient-reported symptoms.

### Specific Objectives
* **Objective 1 (Dataset Characterization & Forensics)**: Conduct an independent forensic audit of the downloaded Kaggle dataset files (`Training.csv` and `Testing.csv`) to resolve documented discrepancies in row counts, column structure, class distributions, and feature redundancy.
* **Objective 2 (Establish Baseline Models)**: Implement a structured hierarchy of conventional machine learning classifiers (Dummy, Naive Bayes, Logistic Regression, Tree Ensembles, SVM, and Multi-Layer Perceptron).
* **Objective 3 (Investigate Generalization & Leakage)**: Empirically contrast the original Kaggle train/test split, standard Stratified Cross-Validation, and Duplicate-Aware / Group-Aware validation protocols.
* **Objective 4 (Evaluate Robustness Under Sparsity & Noise)**: Systematically measure model degradation across varying levels of symptom availability ($132 \to 2$ symptoms) and stochastic reporting noise (false-negative dropout, false-positive activations, combined noise from $0\%$ to $30\%$).
* **Objective 5 (Evaluate Predictive Uncertainty & Calibration)**: Quantify probability calibration using Expected Calibration Error (ECE), Brier Score, and Reliability Diagrams, comparing uncalibrated models against Platt scaling and Isotonic regression.
* **Objective 6 (Explain Predictions & Validate Clinical Associations)**: Apply SHAP to extract global and local feature attributions, evaluate explanation stability under input perturbations, and contextualize identified predictive symptoms against authoritative medical literature.
* **Objective 7 (Build Decision Support Prototype)**: Construct an interactive decision-support interface demonstrating the differential diagnostic outputs, uncertainty indicators, and visual symptom evidence.

---

## 4. Research Questions (RQs) & Hypotheses

### Research Questions
* **RQ1 (Generalization & Leakage)**: *How do different machine-learning algorithms perform in multi-class disease prediction when evaluated using leakage-aware validation strategies compared to the standard Kaggle split?*
* **RQ2 (Symptom Sparsity)**: *How does disease-prediction performance change as the number of available symptoms decreases from complete profiles to sparse chief complaints?*
* **RQ3 (Noise Resilience)**: *How robust are different machine-learning models to false-positive and false-negative symptom reporting under controlled perturbation levels?*
* **RQ4 (Uncertainty Calibration)**: *How accurately do model posterior probability estimates represent true predictive uncertainty in multi-disease symptom classification, and does post-hoc calibration improve reliability?*
* **RQ5 (Explainability & Stability)**: *Which symptoms contribute most strongly to model predictions, and how consistent and stable are these explanations across different diseases, algorithms, and perturbed inputs?*

### Pre-Experiment Hypotheses
* **H1**: Models evaluated using leakage-aware / duplicate-aware validation will demonstrate lower and more realistic generalization performance than models evaluated using the original Kaggle testing split.
* **H2**: Disease-prediction performance will decrease monotonically as the number of available symptoms decreases, but ensemble and calibrated probabilistic models will retain higher Top-3 and Top-5 accuracy than linear baselines.
* **H3**: Increasing symptom-reporting noise will degrade predictive accuracy, with false-positive symptom activations causing steeper degradation than symptom dropouts (false negatives).
* **H4**: Post-hoc probability calibration (Platt scaling / Isotonic regression) will significantly reduce Expected Calibration Error (ECE) and Brier scores across tree-based and neural models.
* **H5**: Tree-based ensembles and neural models will identify disease-specific predictive symptom subsets that remain stable under minor perturbations and align with documented clinical descriptions in medical reference literature.

---

## 5. Phase 0: Dataset Forensics (Experiment 1 — First Priority)

Before training models, a comprehensive **Dataset Forensic Report** must be generated directly from the downloaded CSV files to verify data integrity and document discrepancies:

```
Dataset Forensic Audit (Phase 0)
├── Structural Verification
│   ├── Number of rows (Train vs. Test)
│   ├── Number of columns & header inspection
│   ├── Detection and pruning of artifact columns (e.g., trailing 'Unnamed: 133')
│   └── Feature name standardization (stripping whitespace / special characters)
├── Label & Class Distribution
│   ├── Discrepancy check: 42 diseases stated on Kaggle vs. 41 classes observed
│   ├── Distribution balance in Training.csv (samples per class)
│   └── Distribution in Testing.csv (samples per class)
├── Data Quality & Redundancy Analysis
│   ├── Missing / null values across features and target
│   ├── Duplicate exact rows (features + prognosis)
│   ├── Duplicate feature vectors with identical labels
│   ├── Contradictory records (identical symptom vectors mapped to different diseases)
│   └── Train-Test Overlap (quantifying exact identical rows shared across splits)
└── Output Deliverable
    └── Reproducible Dataset Forensic Report (CSV summary + Markdown documentation)
```

> [!NOTE]
> Academic distinction: Rather than assuming external claims (such as specific duplicate percentages), all data quality metrics will be computed directly through our own reproducible forensic scripts.

---

## 6. Experimental Dataset Splitting Strategy

To directly test **H1** and answer **RQ1**, the benchmarking suite will compare three distinct evaluation protocols:

```
Protocol A: Original Kaggle Split
┌─────────────────────────┐          ┌─────────────────────────┐
│ Training.csv (4,920)    │ ───────► │ Testing.csv (42)        │
└─────────────────────────┘          └─────────────────────────┘
(Reproduces existing literature baseline; evaluates memorization)

Protocol B: Standard Stratified K-Fold CV
┌──────────────────────────────────────────────────────────────┐
│ Combined Dataset (K = 5 or 10 Stratified Folds)              │
│ [Fold 1] [Fold 2] [Fold 3] ... [Fold K]                      │
└──────────────────────────────────────────────────────────────┘
(Standard ML practice; tests if synthetic duplicates cross folds)

Protocol C: Duplicate-Aware / Group-Aware Evaluation
┌──────────────────────────────────────────────────────────────┐
│ Unique Symptom Vector Groups                                 │
│ [Group Vector A] ── All instances kept within one fold       │
│ [Group Vector B] ── All instances kept within one fold       │
│ [Group Vector C] ── All instances kept within one fold       │
└──────────────────────────────────────────────────────────────┘
(Guarantees zero pattern leakage; evaluates true generalization to unseen profiles)
```

---

## 7. Model Hierarchy & Selection

Instead of implementing arbitrary numbers of classifiers, the experimental suite utilizes a structured four-level hierarchy:

### Level 1: Minimum Baselines
* **Dummy Classifier**: Uniform and stratified random baselines establishing theoretical performance floors.
* **Bernoulli / Multinomial Naive Bayes**: Natural probabilistic baseline tailored for binary symptom presence indicators ($x_i \in \{0, 1\}$).
* **Multinomial Logistic Regression**: Regularized linear baseline ($L_1$ for sparsity, $L_2$ for ridge penalty) with direct probabilistic interpretation.

### Level 2: Decision Trees & Ensembles
* **Decision Tree (CART)**: Baseline interpretable tree model; benchmark for maximum depth and split rules.
* **Random Forest**: Primary ensemble benchmark using bootstrap bagging and random feature subspaces.
* **Extra Trees (Extremely Randomized Trees)**: Evaluates variance reduction via random thresholding.
* **Gradient Boosting / XGBoost**: State-of-the-art gradient boosted decision trees optimizing multi-class log-loss (`multi:softprob`).
* *(Optional: LightGBM / CatBoost if performance or latency warrants).*

### Level 3: Kernel Methods
* **Linear SVM**: High-dimensional maximum margin hyperplane.
* **RBF-Kernel SVM**: Non-linear symptom interaction modeling with Platt probability scaling.

### Level 4: Neural Architecture
* **Structured Multi-Layer Perceptron (MLP)**:
  $$\text{Input (132)} \to \text{Dense(128, ReLU)} \to \text{Dropout(0.3)} \to \text{Dense(64, ReLU)} \to \text{Dropout(0.2)} \to \text{Dense}(C, \text{Softmax})$$
  Evaluated with Adam optimizer, early stopping, and categorical cross-entropy.

---

## 8. Feature Selection Experiment

To evaluate whether high diagnostic accuracy requires all 132 symptoms or can be achieved with a compact clinical subset:
* **Feature Subsets**: $\{132, 100, 75, 50, 30, 20, 10\}$ symptoms.
* **Methods Evaluated**:
  1. *Filter Methods*: Mutual Information ($I(X; Y)$) and Chi-Square ($\chi^2$).
  2. *Embedded Methods*: $L_1$-regularized Logistic Regression (Lasso feature elimination) and Random Forest Gini importance.
  3. *Wrapper Methods*: Recursive Feature Elimination (RFE) on the top-performing ensemble.
* **Research Question Answered**: What is the minimal informative symptom subset that maintains $\ge 95\%$ of full-feature macro-F1?

---

## 9. Robustness Experiments: Sparsity & Clinical Noise

### 9.1 Symptom Sparsity Experiment (Cold-Start Evaluation)
Simulates patient encounters where only a limited subset of symptoms is reported:
* **Protocol**: From each disease's true symptom profile, randomly sample subsets of size $k \in \{20, 15, 10, 8, 5, 3, 2\}$ active symptoms.
* **Metrics Tracked**: Macro-F1, Top-1, Top-3, Top-5 Accuracy, and Brier score.
* **Output**: Sparsity Degradation Curves plotting diagnostic accuracy versus available symptom count across models.

### 9.2 Stochastic Noise Experiment (Misreporting Simulation)
* **False-Negative Simulation (Symptom Dropout, $1 \to 0$)**: Patient omits or fails to recognize a symptom with probability $p_{drop} \in \{5\%, 10\%, 15\%, 20\%, 30\%\}$.
* **False-Positive Simulation (Erroneous Reporting, $0 \to 1$)**: Patient incorrectly confirms a symptom with probability $p_{fp} \in \{1\%, 2\%, 5\%, 10\%\}$.
* **Combined Noise**: Simultaneous false negatives and false positives.
* **Rigor Protocol**: All perturbation levels repeated over **5 random seeds** with mean $\pm$ standard deviation reported.

---

## 10. Uncertainty Quantification & Probability Calibration

In a clinical decision-support setting, a prediction score must reflect empirical diagnostic likelihood:
* **Evaluation Metrics**:
  * **Expected Calibration Error (ECE)**: Binned deviation between predicted confidence and empirical accuracy.
  * **Brier Score**: Multi-class mean squared probability error:
    $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C (f_{ic} - y_{ic})^2$$
  * **Reliability Diagrams**: Calibration curves visualizing confidence bins against true fractional accuracy.
* **Calibration Methods Tested**:
  * Raw uncalibrated model probabilities.
  * Sigmoid / Platt Scaling.
  * Isotonic Regression.
* **Differential Diagnostic Ranking**: Evaluating Top-1, Top-3, and Top-5 candidate sets.

---

## 11. Explainable AI (XAI) & Clinical Literature Association

### 11.1 Methodological Rigor
* **Feature Attribution**: SHAP (TreeSHAP for ensembles, KernelSHAP for SVM/MLP) to quantify the positive and negative marginal contributions of each symptom.
* **Local vs. Global Interpretability**:
  * *Global*: Top predictive symptoms per disease across the full cohort.
  * *Local*: Patient-specific waterfall plots detailing supporting symptoms ($+$ SHAP) and missing/opposing symptoms ($-$ SHAP).
* **Explanation Stability Analysis**: Measure whether the top-ranked SHAP features remain stable when minor symptom noise or data subsampling is introduced.

### 11.2 Authoritative Medical Literature Comparison
Rather than claiming an arbitrary "clinical score," the study establishes a qualitative and semi-quantitative review:
* Select representative target diseases (e.g., Dengue, Malaria, Typhoid, GERD, Diabetes, Hepatitis).
* Compare the model's top 5 SHAP-attributed symptoms with diagnostic criteria from authoritative clinical sources (e.g., WHO guidelines, CDC clinical briefs, Harrison's Principles of Internal Medicine).
* Document points of clinical concordance and identify synthetic artifacts where the model relies on non-standard statistical associations.

---

## 12. Decision Support System Prototype

### 12.1 Responsible Presentation Paradigm
The user interface explicitly avoids prescriptive statements (e.g., *"You have Disease X"*). Instead, it presents a transparent decision-support summary:

```
────────────────────────────────────────────────────────────────
                  Disease Prediction Result
────────────────────────────────────────────────────────────────
Primary Condition Candidate:  Dengue
Estimated Model Probability:  72% (Calibrated)
Model Confidence Level:       Moderate

Differential Diagnostic Candidates:
  • Malaria:     14%
  • Typhoid:      7%
  • Other:        7%

Symptom Evidence Breakdown:
  ✓ Supporting Symptoms:     High Fever (+0.28), Headache (+0.21), Joint Pain (+0.18)
  ✗ Non-Reported / Missing:  Vomiting (-0.05)

Safety Notice:
This system provides machine-learning-based decision support for
research purposes and does not replace professional medical consultation.
────────────────────────────────────────────────────────────────
```

### 12.2 Technical Prototype Architecture
* **Backend**: FastAPI (Python) serving model inference, probability calibration, and SHAP explanation endpoints.
* **Frontend**: Modern, responsive web interface (React / Next.js or Jinja2/Tailwind).
* **Extension (Optional Experiment E9)**: Adaptive Symptom Questioning module using information gain to query the next most discriminative symptom.

---

## 13. Master's Thesis Structure (8-Chapter Outline)

* **Chapter 1: Introduction**
  * 1.1 Background & Motivation in Clinical Decision Support
  * 1.2 Problem Statement & Research Gaps
  * 1.3 Research Questions & Measurable Hypotheses
  * 1.4 Research Contributions
  * 1.5 Scope, Boundaries & Thesis Organization
* **Chapter 2: Literature Review**
  * 2.1 Machine Learning in Medical Diagnosis
  * 2.2 Symptom-Based Classification Systems
  * 2.3 Classical Machine Learning & Tree Ensembles
  * 2.4 Deep Tabular Learning Architectures
  * 2.5 Explainable AI (XAI) in Clinical Healthcare
  * 2.6 Uncertainty Quantification & Calibration in Medicine
  * 2.7 Methodological Review of Existing Studies on the Kaggle Disease Dataset
  * 2.8 Identification of Unresolved Research Gaps
* **Chapter 3: Dataset Forensics & Research Methodology**
  * 3.1 Kaggle Dataset Provenance & Structural Characteristics
  * 3.2 Dataset Forensic Analysis & Anomaly Detection
  * 3.3 Duplicate Pattern Analysis & Train-Test Leakage Audit
  * 3.4 Leakage-Aware Splitting & Evaluation Protocols
  * 3.5 Feature Selection Methodologies
  * 3.6 Experimental Framework Design
* **Chapter 4: Algorithmic Modeling & Benchmarking**
  * 4.1 Model Formulations (Baselines, Tree Ensembles, SVM, MLP)
  * 4.2 Hyperparameter Tuning Strategy
  * 4.3 Benchmark Results: Kaggle Split vs. Stratified CV vs. Duplicate-Aware Evaluation
  * 4.4 Feature Selection Experimental Results (132 down to 10 features)
* **Chapter 5: Robustness Under Clinical Degradation**
  * 5.1 Symptom Sparsity Analysis (Performance vs. Available Symptom Count)
  * 5.2 Stochastic False-Negative Noise (Symptom Dropouts)
  * 5.3 Stochastic False-Positive Noise (Erroneous Reporting)
  * 5.4 Combined Noise Degradation & Algorithmic Resilience Comparison
* **Chapter 6: Uncertainty Quantification & Explainability**
  * 6.1 Probability Calibration Analysis (ECE, Brier Score, Reliability Curves)
  * 6.2 Top-k Differential Diagnostic Candidate Evaluation
  * 6.3 Global and Local SHAP Feature Attribution
  * 6.4 Explanation Stability Under Input Perturbations
  * 6.5 Medical Literature Comparison & Clinical Concordance
* **Chapter 7: Clinical Decision Support System Prototype**
  * 7.1 System Requirements & Ethical Design Constraints
  * 7.2 Architecture & API Implementation
  * 7.3 User Interface & Differential Diagnostic Visualizations
  * 7.4 System Testing, Response Latency & Usability Evaluation
* **Chapter 8: Results Synthesis, Discussion & Conclusion**
  * 8.1 Summary of Experimental Findings & Hypothesis Testing
  * 8.2 Answering the Research Questions (RQ1–RQ5)
  * 8.3 Contextual Comparison with Prior Published Literature
  * 8.4 Scientific Limitations & Clinical Generalizability Constraints
  * 8.5 Ethical Considerations in Healthcare AI
  * 8.6 Conclusion & Future Research Directions

---

## 14. Experimental Suite Summary (8 Core Experiments)

| ID | Focus Question | Primary Metrics |
| :--- | :--- | :--- |
| **E1** | What are the exact empirical characteristics, duplicates, and discrepancies in the dataset files? | Row counts, column audit, duplicate counts, overlap % |
| **E2** | How do baseline and conventional ML models perform under standard conditions? | Accuracy, Macro-F1, Precision, Recall, Confusion Matrix |
| **E3** | How does train-test leakage and duplication affect reported model performance? | Comparison of Protocol A (Kaggle), Protocol B (CV), Protocol C (Group) |
| **E4** | Can feature selection identify a compact symptom subset with minimal loss of accuracy? | Performance curves across $\{132, 100, 75, 50, 30, 20, 10\}$ features |
| **E5** | How do models degrade when symptoms are sparse (cold-start chief complaints)? | Macro-F1, Top-1/3/5 accuracy across $k \in \{20, 15, 10, 8, 5, 3, 2\}$ |
| **E6** | How tolerant are models to false-positive and false-negative symptom reporting? | Degradation curves under noise levels $\{0\%, 5\%, 10\%, 15\%, 20\%, 30\%\}$ |
| **E7** | Are model confidence estimates calibrated, and does calibration improve reliability? | ECE, Brier Score, Reliability Diagrams (pre- and post-calibration) |
| **E8** | Can model predictions be explained using SHAP, are explanations stable, and do they align with medical literature? | SHAP waterfall/summary plots, stability index, clinical reference review |
| *E9 (Opt)*| *Can adaptive symptom questioning reduce the questions required for diagnostic confidence?* | *Average queries to threshold, mutual information reduction* |

---

## 15. Six-Month Master's Research Roadmap

```
Month 1: Literature Review & Dataset Forensics
├── In-depth literature review of ML on symptom datasets & published Kaggle studies
├── Execution of Experiment 1: Dataset Forensics script & detailed report
└── Formalization of experimental protocols & hypothesis registry

Month 2: Preprocessing & Leakage-Aware Benchmarking
├── Implementation of data loaders & leakage-aware splitting pipelines
├── Execution of Experiment 2 (Baselines) & Experiment 3 (Leakage comparison)
└── Initial comparative performance tables & cross-validation logs

Month 3: Feature Selection & Robustness Experiments
├── Execution of Experiment 4 (Feature selection curves: 132 down to 10)
├── Execution of Experiment 5 (Symptom sparsity cold-start analysis)
└── Execution of Experiment 6 (Stochastic noise simulations with multiple seeds)

Month 4: Uncertainty, Calibration & Explainability (XAI)
├── Execution of Experiment 7 (ECE, Brier Score, Platt/Isotonic calibration)
├── Execution of Experiment 8 (Global/Local SHAP, stability analysis)
└── Qualitative and semi-quantitative comparison with authoritative medical sources

Month 5: Decision Support Prototype & System Integration
├── Development of Python ML service (FastAPI) and clean web interface
├── Integration of differential diagnosis cards, calibrated gauges & SHAP plots
└── Usability, latency, and API verification

Month 6: Thesis Synthesis, Paper Writing & Defense Preparation
├── Compilation of thesis chapters (Chapters 1–8)
├── Drafting peer-reviewed conference/journal manuscript
└── Thesis defense slide deck and mock presentation
```

---

## 16. Scientific Limitations & Ethical Guardrails

1. **Dataset Provenance**: The dataset is a rule-based, symptom-to-disease classification benchmark rather than a prospective, institutional clinical patient cohort. Clinical sampling provenance is acknowledged as a scientific limitation.
2. **Defensible Academic Framing**: The thesis will not claim clinical autonomy or replace physician diagnostic judgment. It is positioned as an empirical investigation into algorithmic behavior, robustness, uncertainty calibration, and interpretability under controlled symptom perturbations.
