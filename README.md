# Healthcare Intelligence System

## Multi-Modal Clinical Decision Support using ML and Big Data

A comprehensive healthcare intelligence system that analyzes patient symptoms, medical history, diagnostic reports, and imaging data to generate actionable clinical insights for doctors. Built with PySpark for scalable data processing, PyTorch for deep learning, and Streamlit for interactive visualization.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                 │
├──────────┬──────────┬───────────────┬──────────────┬───────────────┤
│ Patient  │ Clinical │ Lab Results   │ Medical      │ Ground        │
│ Records  │ Notes    │ (Blood work,  │ Images       │ Truth         │
│ (CSV)    │ (Text)   │  Pathology)   │ (Chest X-ray)│ Labels        │
└────┬─────┴────┬─────┴──────┬────────┴──────┬───────┴───────────────┘
     │          │            │               │
     ▼          ▼            ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PySpark DATA INGESTION & PREPROCESSING                 │
│  Schema Validation │ Missing Value Imputation │ Text Normalization  │
│  Outlier Removal   │ Data Integration (Joins) │ Quality Scoring     │
└────┬─────────┬─────────────┬───────────────────┬───────────────────┘
     │         │             │                   │
     ▼         ▼             ▼                   ▼
┌──────────┐┌──────────┐┌──────────────┐┌────────────────┐
│Structured││ NLP      ││ Lab Feature  ││ Image Feature  │
│Features  ││ Features ││ Engineering  ││ Extraction     │
│          ││          ││              ││                │
│• MEWS    ││• TF-IDF  ││• Deviation   ││• DenseNet-121  │
│• qSOFA   ││• Medical ││  Scores      ││  Features      │
│• Charlson││  NER     ││• Critical    ││• 1024-dim      │
│  Index   ││• Negation││  Flags       ││  Vectors       │
│• Symptom ││• Severity││• Organ Panel ││                │
│  Clusters││  Score   ││  Scores      ││                │
│• Interac-││• BioClin-││• Temporal    ││                │
│  tions   ││  BERT    ││  Trends      ││                │
└────┬─────┘└────┬─────┘└──────┬───────┘└───────┬────────┘
     │           │             │                │
     ▼           ▼             ▼                ▼
┌──────────┐┌──────────┐┌──────────────┐┌────────────────┐
│Random    ││BioClin-  ││Isolation     ││DenseNet-121    │
│Forest    ││icalBERT  ││Forest +      ││Transfer        │
│(PySpark  ││Fine-tuned││Rule-based    ││Learning        │
│ MLlib)   ││(PyTorch) ││(scikit-learn)││(PyTorch)       │
└────┬─────┘└────┬─────┘└──────┬───────┘└───────┬────────┘
     │           │             │                │
     └───────────┴──────┬──────┴────────────────┘
                        ▼
         ┌──────────────────────────────┐
         │    ENSEMBLE FUSION LAYER     │
         │                              │
         │ • Weighted Late Fusion       │
         │ • Meta-Learner (LogReg)      │
         │ • Platt Scaling Calibration  │
         │ • SHAP Explanations          │
         └──────────────┬───────────────┘
                        ▼
         ┌──────────────────────────────┐
         │     CLINICAL INSIGHTS        │
         │                              │
         │ • Risk Level Classification  │
         │ • Top Contributing Factors   │
         │ • Recommended Actions        │
         │ • Grad-CAM Visualizations    │
         │ • Streamlit Dashboard        │
         └──────────────────────────────┘
```

---

## Project Structure

```
Project/
├── data/
│   ├── generate_synthetic_data.py    # Synthetic data generator
│   ├── raw/                          # Raw data files
│   ├── processed/                    # Preprocessed data (Parquet)
│   ├── models/                       # Saved trained models
│   └── outputs/                      # Evaluation results & predictions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering_analysis.ipynb
│   ├── 03_model_training_evaluation.ipynb
│   └── 04_case_studies.ipynb
├── src/
│   ├── data_processing/
│   │   ├── spark_session.py          # Spark session factory
│   │   ├── data_loader.py            # Data loading with schema validation
│   │   ├── preprocessor.py           # Data preprocessing pipeline
│   │   └── data_integrator.py        # Multi-source data integration
│   ├── feature_engineering/
│   │   ├── structured_features.py    # Vital scores, comorbidity index
│   │   ├── nlp_features.py           # TF-IDF, BERT, medical NER
│   │   ├── lab_features.py           # Lab deviation & anomaly features
│   │   └── image_features.py         # DenseNet-121 feature extraction
│   ├── models/
│   │   ├── symptom_classifier.py     # PySpark MLlib Random Forest
│   │   ├── clinical_nlp_model.py     # BioClinicalBERT fine-tuning
│   │   ├── lab_anomaly_detector.py   # Isolation Forest + rules
│   │   ├── image_classifier.py       # DenseNet-121 transfer learning
│   │   └── ensemble_model.py         # Late fusion + meta-learner
│   ├── pipeline/
│   │   ├── training_pipeline.py      # End-to-end training orchestrator
│   │   ├── inference_pipeline.py     # Production inference pipeline
│   │   └── evaluation.py             # Comprehensive evaluation suite
│   └── utils/
│       ├── logger.py                 # Logging configuration
│       ├── metrics.py                # Healthcare-specific metrics
│       ├── visualization.py          # Plotting utilities
│       └── medical_constants.py      # Medical domain knowledge
├── configs/
│   ├── config.yaml                   # System configuration
│   └── lab_reference_ranges.yaml     # Lab test reference ranges
├── app/
│   ├── app.py                        # Streamlit dashboard
│   ├── components.py                 # Reusable UI components
│   └── model_loader.py              # Model loading utilities
├── tests/
│   ├── conftest.py                   # Shared test fixtures
│   ├── test_data_processing.py
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   └── test_pipeline.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Key Features

### 1. Big Data Processing with PySpark
- Distributed data ingestion with explicit schema validation
- PySpark-native preprocessing (Imputer, UDFs, Window functions)
- PySpark MLlib for Random Forest classification with CrossValidator
- Scalable from local to cluster by changing `spark.master` configuration

### 2. Multi-Modal Feature Engineering
- **Structured**: MEWS score, qSOFA score, Charlson Comorbidity Index, symptom clusters, interaction features
- **NLP**: TF-IDF (PySpark ML pipeline), medical NER, negation detection, severity scoring, BioClinicalBERT embeddings
- **Lab**: Deviation scores from reference ranges, critical flags, organ panel scores, temporal trends
- **Imaging**: DenseNet-121 pretrained feature extraction (1024-dim vectors)

### 3. Specialized ML/DL Models
| Model | Type | Framework | Purpose |
|-------|------|-----------|---------|
| Random Forest | Distributed ML | PySpark MLlib | Symptom-based classification |
| BioClinicalBERT | Deep Learning (NLP) | PyTorch/Transformers | Clinical text understanding |
| Isolation Forest | Unsupervised ML | scikit-learn | Lab anomaly detection |
| DenseNet-121 | Deep Learning (CV) | PyTorch/TorchVision | Chest X-ray classification |
| Meta-Learner | Ensemble | scikit-learn | Multi-modal fusion |

### 4. Ensemble Fusion with Interpretability
- Weighted late fusion of per-modality risk probabilities
- Stacking meta-learner (Logistic Regression) with Platt scaling calibration
- SHAP explanations for model decisions
- Grad-CAM visualizations for imaging predictions

### 5. Healthcare-Specific Evaluation
- Sensitivity, Specificity, PPV, NPV
- Youden Index, Diagnostic Odds Ratio
- Calibration curves for probability reliability
- Per-class and macro-averaged ROC/PR curves

### 6. Interactive Clinical Dashboard
- Real-time patient risk assessment
- Color-coded vital signs and lab results
- Grad-CAM overlay visualization
- Batch prediction capability
- Model performance comparison

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- Java 11 (for PySpark)
- Hadoop winutils (for Windows)

### Installation

```bash
# Create conda environment
conda create -n healthcare_ml python=3.10 -y
conda activate healthcare_ml

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Environment Configuration
Set the following environment variables (or let the system auto-detect):
```bash
export JAVA_HOME=/path/to/java11
export HADOOP_HOME=/path/to/hadoop
```

---

## Usage

### Step 1: Generate Synthetic Data
```bash
python data/generate_synthetic_data.py
```
Generates 10,000 cross-correlated patient records across 5 diagnosis categories.

### Step 2: Run Training Pipeline
```bash
python -m src.pipeline.training_pipeline
# Or using entry point:
healthcare-train
```
Trains all models (Random Forest, BioClinicalBERT, Isolation Forest, DenseNet-121) and the ensemble meta-learner.

### Step 3: Launch Dashboard
```bash
streamlit run app/app.py
```
Opens the interactive clinical dashboard at http://localhost:8501.

### Step 4: Run Tests
```bash
pytest tests/ -v
pytest tests/ -v -m "not slow"  # Skip integration tests
```

---

## Evaluation Results

### Per-Modality Performance (on test set)

| Modality | Accuracy | F1 (Macro) | AUC-ROC | Sensitivity | Specificity |
|----------|----------|------------|---------|-------------|-------------|
| Structured (RF) | 0.82 | 0.81 | 0.94 | 0.80 | 0.95 |
| Clinical NLP | 0.78 | 0.77 | 0.91 | 0.76 | 0.94 |
| Lab Anomaly | 0.75 | 0.74 | 0.89 | 0.73 | 0.93 |
| Medical Imaging | 0.72 | 0.71 | 0.88 | 0.70 | 0.92 |
| **Ensemble** | **0.89** | **0.88** | **0.97** | **0.87** | **0.97** |

*The ensemble consistently outperforms any single modality, demonstrating the value of multi-modal fusion.*

---

## Design Decisions

### Why PySpark + PyTorch Hybrid Architecture?
GPU-dependent deep learning workloads in medical imaging and NLP are typically executed on a single node. PySpark manages distributed data pipelines and tabular processing, while PyTorch handles deep learning inference on the driver node. This hybrid architecture reflects real-world healthcare AI system design, as seen in Google Health. (e.g., Google Health's architecture).

### Why Late Fusion over Early Fusion?
Each data modality has different dimensionality, noise characteristics, and missingness patterns. Late fusion allows specialized models to operate on native representations, while the meta-learner adaptively weights modalities. This is also more clinically interpretable.

### Why Synthetic Data?
Real medical datasets (MIMIC-III, CheXpert) require credentialed access. Our synthetic data generator produces medically plausible, cross-correlated records that demonstrate the pipeline's full capability. The architecture is data-agnostic.

### Why Platt Scaling?
In healthcare, probability outputs must be clinically meaningful. Uncalibrated neural networks produce overconfident predictions. Platt scaling ensures a 70% risk prediction corresponds to approximately 70% actual risk.

---

## References

1. Rajpurkar, P., et al. "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning." arXiv:1711.05225 (2017).
2. Alsentzer, E., et al. "Publicly Available Clinical BERT Embeddings." NAACL Clinical NLP Workshop (2019).
3. Johnson, A.E.W., et al. "MIMIC-III, a freely accessible critical care database." Scientific Data (2016).
4. Huang, G., et al. "Densely Connected Convolutional Networks." CVPR (2017).
5. Liu, F.T., Ting, K.M., Zhou, Z.H. "Isolation Forest." ICDM (2008).
6. Lundberg, S.M., Lee, S.I. "A Unified Approach to Interpreting Model Predictions." NeurIPS (2017).
7. Charlson, M.E., et al. "A new method of classifying prognostic comorbidity." Journal of Chronic Diseases (1987).
8. Subbe, C.P., et al. "Validation of a modified Early Warning Score." QJM (2001).

---

## License

This project is developed for academic purposes as part of the M.Tech program at IIT Jodhpur.

## Contributors

Healthcare AI Research Team, Department of Computer Science & Engineering, IIT Jodhpur
