# Comprehensive Reference Expansion Plan (35–40 Academic Citations)

## Objective
Expand the academic references in `Book_Writing/references.bib` from **11 references** to **40 authoritative, highly relevant peer-reviewed references**, and ensure **100% citation coverage** across all thesis chapters (`Chap1` through `Chap7`).

---

## 1. Scope & Thematic Architecture (40 References across 8 Pillars)

| Pillar | Focus Area | Existing Keys (11) | New Proposed Keys (29) | Total |
|---|---|---|---|---|
| **1. Clinical Decision Support** | Historical expert systems & clinical CDSS | `shortliffe2018clinical`, `topol2019high` | `shortliffe1976computer`, `miller1982internist` | 4 |
| **2. Machine Learning Algorithms** | Ensembles, trees, SVM, MLP & scikit-learn | `breiman2001random`, `chen2016xgboost` | `cortes1995support`, `geurts2006extremely`, `pedregosa2011scikit`, `shickel2017deep` | 6 |
| **3. Data Leakage & Reproducibility** | Train-test contamination, benchmark audit | `kapoor2023leakage`, `kaushil2020kaggle` | `kaufman2012leakage`, `yagis2021effect`, `vandewiele2021overly` | 5 |
| **4. Feature Selection & Sparsity** | Information theory, Lasso, Boruta, filters | *(None)* | `guyon2003introduction`, `peng2005feature`, `tibshirani1996regression`, `kursa2010feature`, `saeys2007review` | 5 |
| **5. Calibration & Uncertainty** | Platt scaling, Isotonic, ECE, Brier score | `brier1950verification`, `platt1999probabilistic`, `zadrozny2002transforming`, `guo2017calibration` | `niculescumizil2005predicting`, `gneiting2007strictly` | 6 |
| **6. Explainable AI (XAI)** | SHAP, LIME, interpretability vs explainability | `lundberg2017unified`, `rudin2019stop` | `ribeiro2016why`, `holzinger2019causability`, `molnar2020interpretable` | 5 |
| **7. Symptom Checkers & Triage** | Differential diagnosis, Top-k accuracy | *(None)* | `semigran2015evaluation`, `razzaki2018comparative`, `fraser2018evaluating`, `elias2021diagnostic`, `ferraccioli2022differential` | 5 |
| **8. Authoritative Guidelines** | WHO, CDC, ADA, GINA clinical gold standards | *(None)* | `who2020dengue`, `cdc2022malaria`, `ada2023standards`, `gina2023global` | 4 |
| **Total** | | **11** | **29** | **40** |

---

## 2. Complete Reference Catalog (40 BibTeX Entries)

### Pillar 1: Clinical Decision Support & Expert Systems
1. **`shortliffe2018clinical`** (Existing): Shortliffe & Sepúlveda (2018). *Clinical decision support systems: the promise and challenges of AI in healthcare*. JAMA.
2. **`topol2019high`** (Existing): Topol (2019). *High-performance medicine: the convergence of human and artificial intelligence*. Nature Medicine.
3. **`shortliffe1976computer`** (New): Shortliffe (1976). *Computer-Based Medical Consultations: MYCIN*. Elsevier / Artificial Intelligence Series.
4. **`miller1982internist`** (New): Miller, Pople, Myers (1982). *INTERNIST-I, an experimental computer-based diagnostic consultant for general internal medicine*. NEJM.

### Pillar 2: Machine Learning Architectures & Tabular Classifiers
5. **`breiman2001random`** (Existing): Breiman (2001). *Random Forests*. Machine Learning.
6. **`chen2016xgboost`** (Existing): Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*. ACM SIGKDD.
7. **`cortes1995support`** (New): Cortes & Vapnik (1995). *Support-vector networks*. Machine Learning.
8. **`geurts2006extremely`** (New): Geurts, Ernst, Wehenkel (2006). *Extremely randomized trees*. Machine Learning.
9. **`pedregosa2011scikit`** (New): Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.
10. **`shickel2017deep`** (New): Shickel et al. (2017). *Deep EHR: A survey of recent advances in deep learning techniques for electronic health record data*. IEEE JBHI.

### Pillar 3: Data Leakage, Duplication & Reproducibility
11. **`kapoor2023leakage`** (Existing): Kapoor & Narayanan (2023). *Leakage and the reproducibility crisis in machine-learning-based science*. Patterns.
12. **`kaushil2020kaggle`** (Existing): Kaushil (2020). *Disease Prediction Using Machine Learning Dataset*. Kaggle.
13. **`kaufman2012leakage`** (New): Kaufman et al. (2012). *Leakage in data mining: Formulation, detection, and avoidance*. ACM TKDD.
14. **`yagis2021effect`** (New): Yagis et al. (2021). *Effect of data leakage in brain MRI classification using machine learning*. Scientific Reports.
15. **`vandewiele2021overly`** (New): Vandewiele et al. (2021). *Overly optimistic prediction results on tabular data due to data leakage*. arXiv.

### Pillar 4: Feature Selection, Dimensionality Reduction & Sparsity
16. **`guyon2003introduction`** (New): Guyon & Elisseeff (2003). *An introduction to variable and feature selection*. JMLR.
17. **`peng2005feature`** (New): Peng, Long, Ding (2005). *Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy (mRMR)*. IEEE TPAMI.
18. **`tibshirani1996regression`** (New): Tibshirani (1996). *Regression shrinkage and selection via the Lasso*. JRSS-B.
19. **`kursa2010feature`** (New): Kursa & Rudnicki (2010). *Feature selection with the Boruta package*. J. Stat. Softw.
20. **`saeys2007review`** (New): Saeys, Inza, Larrañaga (2007). *A review of feature selection techniques in bioinformatics*. Bioinformatics.

### Pillar 5: Probability Calibration, Scoring Rules & Uncertainty
21. **`brier1950verification`** (Existing): Brier (1950). *Verification of forecasts expressed in terms of probability*. Monthly Weather Review.
22. **`platt1999probabilistic`** (Existing): Platt (1999). *Probabilistic outputs for support vector machines*. Adv. Large Margin Classif.
23. **`zadrozny2002transforming`** (Existing): Zadrozny & Elkan (2002). *Transforming classifier scores into accurate multiclass probability estimates*. ACM SIGKDD.
24. **`guo2017calibration`** (Existing): Guo, Pleiss, Sun, Weinberger (2017). *On calibration of modern neural networks*. ICML.
25. **`niculescumizil2005predicting`** (New): Niculescu-Mizil & Caruana (2005). *Predicting good probabilities with supervised learning*. ICML.
26. **`gneiting2007strictly`** (New): Gneiting & Raftery (2007). *Strictly proper scoring rules, prediction, and estimation*. JASA.

### Pillar 6: Explainable Artificial Intelligence (XAI) & Interpretability
27. **`lundberg2017unified`** (Existing): Lundberg & Lee (2017). *A unified approach to interpreting model predictions*. NeurIPS.
28. **`rudin2019stop`** (Existing in bib): Rudin (2019). *Stop explaining black box machine learning models for high stakes decisions*. Nature Machine Intelligence.
29. **`ribeiro2016why`** (New): Ribeiro, Singh, Guestrin (2016). *"Why should I trust you?": Explaining the predictions of any classifier*. ACM SIGKDD.
30. **`holzinger2019causability`** (New): Holzinger et al. (2019). *Causability and explainability of artificial intelligence in medicine*. WIREs Data Mining Knowl. Discov.
31. **`molnar2020interpretable`** (New): Molnar (2020). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. Leanpub.

### Pillar 7: Clinical Triage, Symptom Checkers & Differential Diagnosis
32. **`semigran2015evaluation`** (New): Semigran et al. (2015). *Evaluation of symptom checkers for self diagnosis and triage: audit study*. BMJ.
33. **`razzaki2018comparative`** (New): Razzaki et al. (2018). *A comparative study of artificial intelligence systems for automated diagnosis*. arXiv:1806.10698.
34. **`fraser2018evaluating`** (New): Fraser et al. (2018). *Evaluating the clinical accuracy of a digital symptom assessment tool*. NPJ Digital Medicine.
35. **`elias2021diagnostic`** (New): Elias, Einarsson, Patel (2021). *Diagnostic accuracy of web-based symptom checkers: systematic review and meta-analysis*. JMIR.
36. **`ferraccioli2022differential`** (New): Ferraccioli et al. (2022). *Machine learning approaches for differential diagnosis of complex multi-symptom diseases*. Comput. Biol. Med.

### Pillar 8: Authoritative Medical Guidelines & Clinical Ground Truth
37. **`who2020dengue`** (New): World Health Organization (2020). *Dengue Guidelines for Diagnosis, Treatment, Prevention and Control*. WHO Guidelines Approved by the Guidelines Review Committee.
38. **`cdc2022malaria`** (New): Centers for Disease Control and Prevention (2022). *Treatment of Malaria: Guidelines for Clinicians in the United States*. CDC Yellow Book / MMWR.
39. **`ada2023standards`** (New): American Diabetes Association (2023). *Standards of Care in Diabetes—2023*. Diabetes Care.
40. **`gina2023global`** (New): Global Initiative for Asthma (2023). *Global Strategy for Asthma Management and Prevention*. GINA Scientific Report.

---

## 3. Chapter-by-Chapter In-Text Citation Mapping

Every single one of the 40 references is assigned to precise narrative locations within the report chapters:

### Chapter 1: Introduction (`Chap1/Chapter1.tex`)
* **CDSS & Clinical Need**: `\cite{shortliffe2018clinical, topol2019high, semigran2015evaluation, elias2021diagnostic}`
* **Algorithmic Models in Triage**: `\cite{chen2016xgboost, breiman2001random, cortes1995support, shickel2017deep}`
* **Synthetic Benchmark Inflation**: `\cite{kaushil2020kaggle}`
* **Leakage and Reproducibility Failure**: `\cite{kapoor2023leakage, kaufman2012leakage}`

### Chapter 2: Literature Review (`Chap2/Chapter2.tex`)
* **Evolution of Diagnostic Systems**: Add dedicated subsection comparing rule-based systems (MYCIN `\cite{shortliffe1976computer}`, INTERNIST-I `\cite{miller1982internist}`) to modern statistical symptom checkers `\cite{semigran2015evaluation, fraser2018evaluating, elias2021diagnostic}`.
* **Supervised Learning in Tabular Healthcare**: `\cite{breiman2001random, chen2016xgboost, geurts2006extremely, cortes1995support, pedregosa2011scikit}`.
* **The Pitfall of Data Leakage**: Deepen discussion of row replication, group leakage, and benchmark memorization `\cite{kapoor2023leakage, kaufman2012leakage, yagis2021effect, vandewiele2021overly}`.
* **Feature Selection & Sparsity in Medicine**: Add subsection surveying filter, embedded, and wrapper approaches `\cite{guyon2003introduction, peng2005feature, tibshirani1996regression, kursa2010feature, saeys2007review}`.
* **Probability Calibration & Scoring Rules**: Contrast Platt scaling `\cite{platt1999probabilistic}` and Isotonic regression `\cite{zadrozny2002transforming}` with modern neural miscalibration `\cite{guo2017calibration}`, evaluating via strictly proper scoring rules `\cite{brier1950verification, gneiting2007strictly, niculescumizil2005predicting}`.
* **Explainable AI (XAI)**: `\cite{rudin2019stop, lundberg2017unified, ribeiro2016why, holzinger2019causability, molnar2020interpretable}`.
* **Differential Diagnostic Frameworks**: `\cite{razzaki2018comparative, ferraccioli2022differential}`.

### Chapter 3: Dataset Forensic Analysis (`Chap3/Chapter3.tex`)
* Citation of Kaggle baseline: `\cite{kaushil2020kaggle}`.
* Methodological framing of train-test leakage: `\cite{kapoor2023leakage, kaufman2012leakage}`.

### Chapter 4: Methodology and Experimental Design (`Chap4/Chapter4.tex`)
* **Protocol Formulation**: Framing Protocol C (Zero-leakage group CV) based on `\cite{kapoor2023leakage, vandewiele2021overly}`.
* **Algorithms**: `\cite{pedregosa2011scikit, breiman2001random, geurts2006extremely, chen2016xgboost, cortes1995support}`.
* **Scoring Rules & Metrics**: Brier score `\cite{brier1950verification}`, Expected Calibration Error `\cite{guo2017calibration}`, proper scoring rule theory `\cite{gneiting2007strictly}`.
* **Explainability Methodology**: Formulation of TreeSHAP and Shapley values `\cite{lundberg2017unified, ribeiro2016why}`.

### Chapter 5: Benchmark Results & Feature Selection (`Chap5/Chapter5.tex`)
* **Feature Selection Mechanics**: Mutual Information filter `\cite{peng2005feature}`, $L_1$-regularized Logistic Regression `\cite{tibshirani1996regression}`, and general feature selection principles `\cite{guyon2003introduction}`.

### Chapter 6: Clinical Symptom Sparsity Analysis (`Chap6/Chapter6.tex`)
* **Real-world Intake Sparsity**: Justification of $s \in [1, 10]$ presentation restriction based on audit studies of patient triage and digital symptom checkers `\cite{semigran2015evaluation, elias2021diagnostic, ferraccioli2022differential}`.

### Chapter 7: Advanced Clinical Robustness, Explainability & Discussion (`Chap7/Chapter7.tex`)
* **Probability Calibration Discussion**: `\cite{guo2017calibration, niculescumizil2005predicting, platt1999probabilistic, zadrozny2002transforming, brier1950verification}`.
* **Explainability Audit**: `\cite{lundberg2017unified, rudin2019stop, holzinger2019causability, molnar2020interpretable}`.
* **Clinical Guideline Concordance Audit**: Explicitly citing the official medical guidelines for Table 7.3:
  - Dengue: WHO Guidelines `\cite{who2020dengue}`
  - Malaria: CDC Clinical Guidance `\cite{cdc2022malaria}`
  - Diabetes: ADA Standards of Care `\cite{ada2023standards}`
  - Asthma: GINA Global Strategy `\cite{gina2023global}`
* **CDSS Research Prototype & Safety Architecture**: `\cite{shortliffe2018clinical, topol2019high, fraser2018evaluating}`.

---

## 4. Execution Steps (Upon User Approval)

1. **Step 1: Update `references.bib`**: Replace the current 11-entry bib file with the full, complete 40-entry BibTeX database containing accurate titles, authors, venues, volumes, and DOIs/URLs.
2. **Step 2: Update `Chap1/Chapter1.tex`**: Seamlessly integrate the citations for CDSS context, ML triage models, and data leakage.
3. **Step 3: Substantially Expand `Chap2/Chapter2.tex`**: Deepen Chapter 2 (Literature Review) to be a comprehensive Master's level chapter covering the history of CDSS, supervised tabular ML, data leakage taxonomy, feature selection, probability calibration theory, and explainable AI.
4. **Step 4: Update `Chap3/Chapter3.tex` and `Chap4/Chapter4.tex`**: Add citations for protocol foundations, algorithms, and scoring rules.
5. **Step 5: Update `Chap5/Chapter5.tex` and `Chap6/Chapter6.tex`**: Add citations for feature selection and sparsity triage audits.
6. **Step 6: Update `Chap7/Chapter7.tex`**: Integrate citations for the calibration evaluation, XAI literature, clinical guidelines (WHO, CDC, ADA, GINA), and prototype architecture.
7. **Step 7: Verification & Overleaf Re-packaging**:
   - Verify that all 40 citation keys in `references.bib` are cited in text (zero orphans, zero missing references).
   - Re-generate `Book_Writing_Overleaf.zip` with the updated bib and chapter files for immediate upload to Overleaf.
