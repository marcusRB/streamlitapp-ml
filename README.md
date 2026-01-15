# CKD Detection - MLOps Project

A structured MLOps project for Chronic Kidney Disease (CKD) detection using machine learning.

## Project Structure

```
ckd-detection/
│
├── data/
│   ├── raw/                          # Original, immutable data
│   │   └── chronic_kindey_disease.csv
│   └── processed/                    # Cleaned, processed data
│       ├── loaded_data.csv
│       └── cleaned_data.csv
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── data_loading.py               # Data loading module
│   └── data_processing.py            # Data processing module
│
├── reports/                          # Generated analysis reports
│   ├── missing_values_report.csv
│   ├── numeric_summary.csv
│   ├── categorical_summary.csv
│   ├── correlation_matrix.csv
│   └── data_quality_report.json
│
├── figures/                          # Generated visualizations
│   ├── missing_values_heatmap.png
│   ├── numeric_distributions.png
│   ├── target_distribution.png
│   ├── categorical_distributions.png
│   └── correlation_matrix.png
│
├── requirements.txt                  # Python dependencies
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

## Usage

### 1. Data Loading

Run the data loading module to load and validate raw data:

```bash
cd src
python data_loading.py
```

**Output:**
- Validates data schema
- Displays dataset information
- Saves loaded data to `data/processed/loaded_data.csv`
- Logs execution to console

### 2. Data Processing

Run the data processing module to clean, analyze, and visualize data:

```bash
cd src
python data_processing.py
```

**Output:**
- Cleans data (removes whitespace, standardizes values)
- Generates missing value analysis
- Creates statistical summaries
- Generates all visualizations
- Saves cleaned data to `data/processed/cleaned_data.csv`
- Saves reports to `reports/` directory
- Saves figures to `figures/` directory

## Module Details

### data_loading.py

**Purpose:** Handle raw data ingestion and initial validation

**Key Features:**
- Load CSV with custom null identifiers
- Validate data schema
- Generate basic dataset statistics
- Save loaded data for processing

**Main Class:** `DataLoader`

**Key Methods:**
- `load_data()`: Load raw CSV file
- `validate_schema()`: Validate data structure
- `get_data_info()`: Get dataset statistics
- `save_loaded_data()`: Save to processed directory

### data_processing.py

**Purpose:** Clean, analyze, and visualize data

**Key Features:**
- Data cleaning (text normalization, standardization)
- Missing value analysis and visualization
- Statistical summaries (numeric and categorical)
- Univariate analysis (distributions)
- Correlation analysis
- Comprehensive quality reporting

**Main Class:** `DataProcessor`

**Key Methods:**
- `clean_data()`: Clean and standardize data
- `analyze_missing_values()`: Analyze missing data patterns
- `generate_statistical_summary()`: Create summary statistics
- `plot_univariate_numeric()`: Visualize numeric features
- `plot_univariate_categorical()`: Visualize categorical features
- `plot_correlation_matrix()`: Create correlation heatmap
- `generate_data_quality_report()`: Comprehensive quality report

## Dataset Information

**Source:** Alagappa University & Apollo Hospitals, Tamil Nadu, India (July 2015)

**Size:** 400 patient instances, 25 features (24 predictive + 1 target)

**Target:** Binary classification (CKD vs Not CKD)
- CKD cases: ~250
- Control cases: ~150

**Feature Types:**
- Numerical (11): Age, blood pressure, blood chemistry levels
- Categorical (14): Symptoms, test results, medical history

## Logging

Both modules use Python's logging module to track execution:
- INFO level: Normal operations and progress
- WARNING level: Unexpected conditions
- ERROR level: Failures and exceptions

Logs are printed to console with timestamps.

## Next Steps

After running these modules, you can proceed to:
1. Feature engineering
2. Model training
3. Model evaluation
4. Model deployment

## License

This project is for educational purposes.