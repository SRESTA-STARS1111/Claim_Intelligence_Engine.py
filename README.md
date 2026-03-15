# Claim Intelligence Framework (CIF)

The **Claim Intelligence Framework (CIF)** is an end-to-end Machine Learning ecosystem designed to optimize the healthcare revenue cycle by predicting claim outcomes. This framework documents five distinct engineering pillars utilized to solve the problem of claim denials with high precision.

## Technical Pillars and Decision Logic

### 1. Distributed Big Data Foundation (`pyspark_mllib_core.py`)

* **Process:** Integration of a native **PySpark MLlib** pipeline utilizing `VectorAssembler` and `StandardScaler`.
* **Decision:** Native Spark implementation ensures the framework scales from local samples to enterprise-grade datasets.
* **Evaluation:** Application of `randomSplit()` with a specific seed creates reproducible **70/15/15** train, validation, and test sets.

### 2. Scientific EDA and Exploratory Investigation (`healthcare_eda_and_xgboost.py`)

* **Process:** Deep-dive statistical analysis performed using **Pandas** and **Seaborn**.
* **Decision:** Identification of class imbalance justifies the implementation of **SMOTE oversampling** to prevent model bias toward 'Paid' outcomes.
* **Visuals:** Generation of **confusion matrix heatmaps** provides visibility into misclassification patterns for specific denial reasons.

### 3. Domain-Specific Feature Logic (`ordinal_domain_logic_model.py`)

* **Process:** Extraction of **CARC** (Claim Adjustment Reason Code) prefixes (e.g., 'CO', 'PR').
* **Decision:** Mapping codes to industry categories like *'Contractual Obligations'* captures clinical and financial intent.
* **Advanced Math:** Implementation of **Ordinal Regression** via the `mord` library respects the logical hierarchy of outcomes (e.g., *Denied* → *Document Request* → *Paid*).

### 4. Advanced Inference Engine (`claim_intelligence_engine.py`)

* **Process:** Combination of **Frequency Encoding** with **CatBoost** and **SHAP** explainability.
* **Decision:** Selection of CatBoost for native handling of categorical features and **SHAP (SHapley Additive exPlanations)** for auditable predictions.
* **Refinement:** Manual adjustment of **class weights** improves precision for high-priority denial predictions.

### 5. AutoML and Performance Benchmarking (`healthcare_automl_baseline.py`)

* **Process:** Utilization of **PyCaret’s** `setup()` to initialize the environment and detect column types.
* **Decision:** Execution of PyCaret after custom engineering stages establishes a **"North Star" baseline** to validate manual feature engineering performance.
* **Validation:** Use of `compare_models()` and `tune_model()` verifies hyperparameters for the final production model.

---

## Engineering Decision Matrix

| Challenge | Solution | Technical Justification |
| --- | --- | --- |
| **Class Imbalance** | SMOTE & Class Weights | Training set balanced to 650 samples per minority class to prevent bias. |
| **High Cardinality** | Frequency Encoding | Preserves signal without dimensionality explosion for insurance codes. |
| **Data Scalability** | PySpark ML Pipelines | MLlib's Pipeline API ensures a unified, scalable transformation process. |
| **Interpretability** | SHAP TreeExplainer | Ensures model transparency for healthcare finance audits. |
| **Environment** | Docker & Kubernetes | Containerization ensures consistent runtime and dynamic scaling. |
| **Licensing** | Open-Source (Apache 2.0) | Eliminates licensing costs and prevents proprietary vendor lock-in. |

---

##  Requirements

* **Data Engineering:** `pyspark`, `pandas`, `numpy`
* **Modeling:** `scikit-learn`, `catboost`, `xgboost`, `mord`
* **Explainability:** `shap`, `matplotlib`, `seaborn`, `pycaret`

## Legal and Compliance

* **License:** Apache License 2.0.
* **Data Safety:** All logic is demonstrated using a generic `sample_healthcare_dataset` to ensure privacy and compliance.

