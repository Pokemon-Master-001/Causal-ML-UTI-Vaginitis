# From SHAP to causal inference: enhancing interpretability and stability in multi-outcome risk prediction for urinary and vaginal infection

## 📌 Overview
Abstract: Urinary tract infection (UTI) and vaginitis are among the most common infectious diseases affecting women worldwide, often leading to severe ICU complications such as acute kidney injury (AKI), sepsis, and recurrent ICU admissions. Despite their clinical relevance and frequent co-occurrence, existing predictive research remains fragmented—typically focusing on single diseases or isolated outcomes—and rarely considers the shared pathophysiologic mechanisms underlying these infections. Despite their clinical significance, current predictive research remains limited, often focusing on single diseases or isolated outcomes. Conventional models are prone to spurious correlations, fail to capture state-specific causal pathways, and inadequately address the complex interactions of clinical indicators across diverse patient states. To address these challenges, we leveraged electronic health record data—including demographics, diagnoses, vital signs, laboratory results, and medication records—to develop a comprehensive outcome prediction framework for UTI and vaginitis patients. Our approach integrates causal inference techniques, novel imputation methods for missing values, and machine learning and deep learning models to predict multiple adverse outcomes, including AKI, sepsis, mortality, readmission, and recurrent ICU admissions. The proposed framework provides a holistic analysis of patient trajectories, uncovers key predictive factors, identifies high-risk patients, and improves model accuracy and generalizability compared with conventional approaches. It further elucidates shared and distinct patterns of adverse outcomes between UTIs and vaginitis, offering actionable insights for clinical risk stratification. This study demonstrates that integrating causal modeling with multi-modal predictive analytics can enhance precision in multi-outcome prediction for women’s health. Early identification of high-risk patients enables targeted interventions, reducing morbidity and mortality, and advancing precision medicine in the management of UTIs and vaginitis.

This repository implements the full modeling pipeline described in our study, which integrates:

- Causal discovery (CD) for feature selection
- Causal feature set construction for 5 clinical risk outcomes
- Counterfactual matching analysis for UVC vs. non-UVC cohorts
- Six traditional machine-learning models for predictive modeling

The framework is designed for reproducible clinical ML research and leverages routinely collected ICU data from MIMIC-IV & III.

## 🧬 1. Causal Discovery for Feature Selection
We evaluate three representative causal discovery categories:
| Category                          | Method       | Rationale                                           |
| --------------------------------- | ------------ | --------------------------------------------------- |
| **Functional-based**              | DirectLiNGAM | Identifies causal direction through non-Gaussianity |
| **Score-based**                   | GES          | Greedy search optimizing BIC                        |
| **Continuous optimization-based** | CORL         | Differentiable DAG acyclicity constraints           |

All CD algorithms are implemented via gCastle (Huawei Noah’s Ark Lab). Each method runs 5 times. Retain edges appearing in ≥ 3 out of 5 runs. Combine results across DirectLiNGAM, GES, and CORL to derive unified causal feature sets.

## 🧪 2. Predictor Variables
Causal discovery is performed exclusively on ICU-stay data, including:
- Vital signs
- Laboratory measurements
- Dynamically monitored parameters

Baseline characteristics (demographics, diagnoses, procedures, medications) are used as foundational variables for both:
- Propensity-score matching
- Model training with causal features

For each outcome, the final causal feature set includes all variables possessing direct or indirect edges into the outcome node.

## 🎯 3. Construction of Outcome-Specific Causal Feature Sets

For each outcome: Run DirectLiNGAM, GES, and CORL (5× each), Apply majority vote (≥3/5 edges retained), Extract all direct + indirect ancestors of the outcome, Merge features across methods, Save the unified causal feature set. This produces stable, interpretable, and clinically meaningful variable sets reflecting mechanistic pathways.

## 🧬 4. Counterfactual Matching Analysis (UVC vs. Non-UVC)

To understand distinct causal pathways associated with UVC:
- Construct a 1:1 nearest-neighbor propensity-score matched cohort
- Propensity score estimated using logistic regression, based on:
  - Demographics
  - Diagnoses-
  - Procedures

This analysis identifies condition-specific vs. shared causal features, revealing mechanisms underlying adverse outcomes in neonates with UVC.

## 🤖 5. Prediction Models

Six traditional ML models are implemented: Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), AdaBoost, Support Vector Machine (SVM), XGBoost.

These models are widely used in medicine and offer complementary strengths: linear modeling, tree-based non-linearity, boosting, and kernel methods. All models are trained using: Causal feature sets, Full ICU feature sets (benchmarking).

Performance metrics include: AUROC, AUPRC.

# 📁 Repository Structure

```text
project/
│
├─ data/
│   ├─ MIMIC_patients_0.ipynb
│   └─ MIMIC_TS_CHART_LAB.ipynb
│
├─ cd/                       # Causal discovery modules
│   ├─ runner.py
│   └─ algorithms.py         # DirectLiNGAM, GES, CORL wrappers
│
├─ matching/
│   ├─ ps_model.py           # Logistic regression PS model
│   └─ knn_matching.py       # 1:1 nearest neighbor matching
│
├─ features/
│   └─ build_causal_sets.py  # Construct outcome-specific causal features
│
├─ models/
│   ├─ classical.py          # LR, DT, RF, SVM, AdaBoost, XGBoost
│   └─ train_predictor.py    # Train + evaluate 6 ML models
│
└─ main_cd.py                # Run CD for all outcomes, Build causal feature sets, PS-matching analysis,Predictive modeling

