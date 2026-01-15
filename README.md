# CKD Detection - Complete MLOps Project

A comprehensive MLOps project for Chronic Kidney Disease (CKD) detection with end-to-end pipeline from data loading to model deployment.

## Project Structure

```
ckd-detection/
│
├── data/
│   ├── raw/                          # Original, immutable data
│   │   └── chronic_kindey_disease.csv
│   └── processed/                    # Cleaned, processed data
│       ├── loaded_data.csv           # Initially loaded data
│       ├── cleaned_data.csv          # Cleaned data
│       ├── ckd_imputed.csv           # Imputed dataset (for GB models)
│       └── ckd_normalized.csv        # Normalized dataset (for KNN/SVM)
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── app.py                        # Streamlit dashboard
│   ├── data_loading.py               # Data loading module
│   ├── data_processing.py            # Data processing module
│   ├── feature_engineering.py        # Feature engineering module
│   ├── model_training.py             # Model training module
│   └── model_prediction.py           # Model prediction module
│
├── models/                           # Trained models
│   ├── knn_model.pkl
│   ├── svm_model.pkl
│   ├── gb_imputed_model.pkl
│   └── hist_gb_model.pkl
│
├── reports/                          # Generated analysis reports
│   ├── missing_values_report.csv
│   ├── numeric_summary.csv
│   ├── categorical_summary.csv
│   ├── correlation_matrix.csv
│   ├── data_quality_report.json
│   ├── feature_engineering_stats.json
│   └── models/
│       └── model_comparison.json
│
├── figures/                          # Generated visualizations
│   ├── missing_values_heatmap.png
│   ├── numeric_distributions.png
│   ├── target_distribution.png
│   ├── categorical_distributions.png
│   ├── correlation_matrix.png
│   └── models/
│       ├── knn_optimization.png
│       ├── knn_confusion_matrix.png
│       ├── svm_confusion_matrix.png
│       ├── gradientboosting_imputed_confusion_matrix.png
│       └── histgradientboosting_confusion_matrix.png
│
├── requirements.txt                  # Python dependencies
├── run_app.sh                        # Unix launch script
├── run_app.bat                       # Windows launch script
└── README.md                         # This file
```

## Requirements

- Python 3.11
- See `requirements.txt` for package dependencies

## Installation

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Complete Pipeline Usage

### Option 1: Using Streamlit Dashboard (Recommended)

```bash
# Launch dashboard
cd src
streamlit run app.py

# Or use the provided scripts
./run_app.sh         # Unix/Mac
run_app.bat          # Windows
```

The dashboard provides an interactive interface for the entire pipeline:
1. **Data Loading** - Load and validate raw data
2. **Data Processing** - Clean and process data
3. **Feature Engineering** - Create imputed and normalized datasets
4. **Model Training** - Train multiple ML models
5. **Predictions** - Make predictions with trained models
6. **Analysis** - Explore data distributions and statistics

### Option 2: Using Command Line Modules

#### 1. Data Loading
```bash
cd src
python data_loading.py
```
**Output:** `data/processed/loaded_data.csv`

#### 2. Data Processing
```bash
python data_processing.py
```
**Output:** 
- `data/processed/cleaned_data.csv`
- Reports in `reports/`
- Figures in `figures/`

#### 3. Feature Engineering
```bash
python feature_engineering.py
```
**Output:**
- `data/processed/ckd_imputed.csv`
- `data/processed/ckd_normalized.csv`
- `reports/feature_engineering_stats.json`

#### 4. Model Training
```bash
python model_training.py
```
**Output:**
- Models in `models/`
- Confusion matrices in `figures/models/`
- `reports/models/model_comparison.json`

#### 5. Model Prediction
```bash
python model_prediction.py
```
**Output:** Example predictions with all trained models

## Module Details

### 1. data_loading.py
**Purpose:** Load and validate raw data

**Features:**
- CSV loading with custom null handling
- Schema validation
- Basic statistics
- Data persistence

**Main Class:** `DataLoader`

### 2. data_processing.py
**Purpose:** Clean and analyze data

**Features:**
- Text normalization
- Missing value analysis
- Statistical summaries
- Visualizations (distributions, correlations)
- Quality reporting

**Main Class:** `DataProcessor`

### 3. feature_engineering.py
**Purpose:** Prepare features for modeling

**Features:**
- Target encoding (ckd=1, notckd=0)
- Feature selection (9 key features)
- Missing value imputation
- Feature normalization
- Two output datasets:
  - **Imputed:** For Gradient Boosting models
  - **Normalized:** For KNN and SVM

**Main Class:** `FeatureEngineer`

**Selected Features:**
- `hemo` (Hemoglobin)
- `sg` (Specific Gravity)
- `sc` (Serum Creatinine)
- `rbcc` (Red Blood Cell Count)
- `pcv` (Packed Cell Volume)
- `htn` (Hypertension)
- `dm` (Diabetes Mellitus)
- `bp` (Blood Pressure)
- `age` (Age)

### 4. model_training.py
**Purpose:** Train and evaluate ML models

**Features:**
- GridSearchCV for hyperparameter tuning
- Cross-validation
- Multiple models:
  - **KNN** (k=1-30, optimized)
  - **SVM** (RBF/Linear/Sigmoid kernels)
  - **Gradient Boosting** (with imputed data)
  - **Histogram Gradient Boosting** (native NaN handling)
- Comprehensive metrics (Accuracy, Precision, Recall, F1)
- Confusion matrices
- Model persistence

**Main Class:** `ModelTrainer`

### 5. model_prediction.py
**Purpose:** Load models and make predictions

**Features:**
- Load trained models
- Single patient prediction
- Batch prediction
- Probability estimates
- Consensus prediction (ensemble)
- Model information retrieval

**Main Class:** `ModelPredictor`

### 6. app.py
**Purpose:** Interactive Streamlit dashboard

**Features:**
- Complete pipeline integration
- Real-time visualization
- Interactive parameter input
- Progress tracking
- Model comparison
- Prediction interface

## Model Performance

Based on the original notebook, expected performance rankings:

1. **Gradient Boosting (Imputed)** - F1: ~0.99
2. **Histogram Gradient Boosting** - F1: ~0.97
3. **SVM** - F1: ~0.96
4. **KNN** - F1: ~0.93

All models use:
- 80/20 train-test split
- Stratified sampling
- GridSearchCV for optimization
- Cross-validation (5-10 folds)

## Dataset Information

**Source:** Alagappa University & Apollo Hospitals, Tamil Nadu, India (July 2015)

**Size:** 400 patient instances, 25 features (24 predictive + 1 target)

**Target:** Binary classification
- CKD cases: ~250 (62.5%)
- Control cases: ~150 (37.5%)

**Feature Types:**
- **Numerical (11):** Age, blood pressure, blood chemistry
- **Categorical (14):** Symptoms, test results, medical history

## Logging

All modules use Python's logging:
- **INFO:** Normal operations
- **WARNING:** Unexpected conditions
- **ERROR:** Failures

Logs display in console with timestamps.

## API Reference

### Quick Start Example

```python
from data_loading import DataLoader
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from model_prediction import ModelPredictor

# Load and prepare data
loader = DataLoader()
df = loader.load_data()

# Feature engineering
engineer = FeatureEngineer()
df_normalized = engineer.process_pipeline_normalized()

# Train model
trainer = ModelTrainer()
X_train, X_test, y_train, y_test = trainer.load_data('../../data/processed/ckd_normalized.csv')
knn_results = trainer.train_knn(X_train, X_test, y_train, y_test)

# Make predictions
predictor = ModelPredictor()
predictor.load_model('KNN')

patient = {
    'hemo': 15.4, 'sg': 1.020, 'sc': 1.2, 
    'rbcc': 5.2, 'pcv': 44.0, 'htn': 1.0, 
    'dm': 1.0, 'bp': 80.0, 'age': 48.0
}

result = predictor.predict_single('KNN', patient)
print(f"Prediction: {result['prediction']}")
```

## Troubleshooting

### Models not found
```
FileNotFoundError: Model file not found
```
**Solution:** Run `python model_training.py` first

### Data files missing
```
FileNotFoundError: Data file not found
```
**Solution:** Ensure data is in `data/raw/` and run pipeline in order

### Import errors
```
ModuleNotFoundError: No module named 'data_loading'
```
**Solution:** Run from `src/` directory or check Python path

## Contributing

This is an educational project. Feel free to extend with:
- Additional ML models
- Feature importance analysis
- SHAP explanations
- Model deployment (Flask/FastAPI)
- Docker containerization

## License

Educational use only.