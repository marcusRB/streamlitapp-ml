"""
Configuration Module for Backend
Centralized configuration management
"""

from pathlib import Path
from typing import Dict, Any
import os
from dataclasses import dataclass


@dataclass
class Paths:
    """Path configuration"""
    # Root directories
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    BACKEND_ROOT = PROJECT_ROOT / "backend"
    
    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # Model directories
    MODELS_DIR = BACKEND_ROOT / "ml" / "models"
    
    # Output directories
    REPORTS_DIR = PROJECT_ROOT / "reports"
    FIGURES_DIR = PROJECT_ROOT / "figures"
    MLRUNS_DIR = PROJECT_ROOT / "mlruns"
    
    # Data files
    RAW_DATA_FILE = RAW_DATA_DIR / "chronic_kindey_disease.csv"
    LOADED_DATA_FILE = PROCESSED_DATA_DIR / "loaded_data.csv"
    CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_data.csv"
    IMPUTED_DATA_FILE = PROCESSED_DATA_DIR / "ckd_imputed.csv"
    NORMALIZED_DATA_FILE = PROCESSED_DATA_DIR / "ckd_normalized.csv"
    
    # Model files
    KNN_MODEL_FILE = MODELS_DIR / "knn_model.pkl"
    SVM_MODEL_FILE = MODELS_DIR / "svm_model.pkl"
    GB_MODEL_FILE = MODELS_DIR / "gb_imputed_model.pkl"
    HIST_GB_MODEL_FILE = MODELS_DIR / "hist_gb_model.pkl"
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.REPORTS_DIR,
            cls.FIGURES_DIR,
            cls.MLRUNS_DIR,
            cls.FIGURES_DIR / "models",
            cls.REPORTS_DIR / "models"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model configuration"""
    # Feature names
    FEATURE_NAMES = ['hemo', 'sg', 'sc', 'rbcc', 'pcv', 'htn', 'dm', 'bp', 'age']
    
    # Feature descriptions
    FEATURE_DESCRIPTIONS = {
        'hemo': 'Hemoglobin (g/dL)',
        'sg': 'Specific Gravity',
        'sc': 'Serum Creatinine (mg/dL)',
        'rbcc': 'Red Blood Cell Count (millions/cmm)',
        'pcv': 'Packed Cell Volume (%)',
        'htn': 'Hypertension (0=No, 1=Yes)',
        'dm': 'Diabetes Mellitus (0=No, 1=Yes)',
        'bp': 'Blood Pressure (mmHg)',
        'age': 'Age (years)'
    }
    
    # Feature ranges
    FEATURE_RANGES = {
        'hemo': (0.0, 20.0),
        'sg': (1.000, 1.030),
        'sc': (0.0, 20.0),
        'rbcc': (0.0, 10.0),
        'pcv': (0.0, 60.0),
        'htn': (0.0, 1.0),
        'dm': (0.0, 1.0),
        'bp': (0.0, 200.0),
        'age': (0.0, 120.0)
    }
    
    # Target mapping
    TARGET_MAPPING = {'ckd': 1, 'notckd': 0}
    REVERSE_TARGET_MAPPING = {1: 'ckd', 0: 'notckd'}
    
    # Available models
    AVAILABLE_MODELS = {
        'KNN': 'knn_model.pkl',
        'SVM': 'svm_model.pkl',
        'GradientBoosting': 'gb_imputed_model.pkl',
        'HistGradientBoosting': 'hist_gb_model.pkl'
    }
    
    # Model display names
    MODEL_DISPLAY_NAMES = {
        'KNN': 'K-Nearest Neighbors',
        'SVM': 'Support Vector Machine',
        'GradientBoosting': 'Gradient Boosting',
        'HistGradientBoosting': 'Histogram Gradient Boosting'
    }


@dataclass
class APIConfig:
    """API configuration"""
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", 8000))
    RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
    LOG_LEVEL = os.getenv("API_LOG_LEVEL", "info")
    
    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # API metadata
    TITLE = "CKD Detection API"
    DESCRIPTION = "REST API for Chronic Kidney Disease prediction using ML models"
    VERSION = "1.0.0"


@dataclass
class MLflowConfig:
    """MLflow configuration"""
    TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{Paths.MLRUNS_DIR}")
    EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "CKD_Detection")
    INFERENCE_EXPERIMENT_NAME = "CKD_Inference"
    ENABLED = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"


@dataclass
class DataConfig:
    """Data processing configuration"""
    # Missing value handling
    NA_VALUES = ['?']
    IMPUTATION_STRATEGY = 'median'
    
    # Train/test split
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    STRATIFY = True
    
    # Cross-validation
    CV_FOLDS = 10
    CV_SCORING = 'f1'


@dataclass
class LoggingConfig:
    """Logging configuration"""
    LEVEL = os.getenv("LOG_LEVEL", "INFO")
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


class Settings:
    """Main settings class"""
    
    def __init__(self):
        self.paths = Paths()
        self.model = ModelConfig()
        self.api = APIConfig()
        self.mlflow = MLflowConfig()
        self.data = DataConfig()
        self.logging = LoggingConfig()
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path to model file"""
        if model_name not in self.model.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_name}' not found")
        
        filename = self.model.AVAILABLE_MODELS[model_name]
        return self.paths.MODELS_DIR / filename
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'paths': {
                'project_root': str(self.paths.PROJECT_ROOT),
                'data_dir': str(self.paths.DATA_DIR),
                'models_dir': str(self.paths.MODELS_DIR),
                'reports_dir': str(self.paths.REPORTS_DIR),
                'figures_dir': str(self.paths.FIGURES_DIR)
            },
            'model': {
                'features': self.model.FEATURE_NAMES,
                'available_models': list(self.model.AVAILABLE_MODELS.keys())
            },
            'api': {
                'host': self.api.HOST,
                'port': self.api.PORT,
                'title': self.api.TITLE
            },
            'mlflow': {
                'tracking_uri': self.mlflow.TRACKING_URI,
                'experiment': self.mlflow.EXPERIMENT_NAME,
                'enabled': self.mlflow.ENABLED
            }
        }


# Global settings instance
settings = Settings()

# Create directories on import
Paths.create_directories()


if __name__ == "__main__":
    # Test configuration
    print("CKD Detection - Configuration")
    print("=" * 60)
    
    print("\n[Paths]")
    print(f"Project Root: {settings.paths.PROJECT_ROOT}")
    print(f"Data Dir: {settings.paths.DATA_DIR}")
    print(f"Models Dir: {settings.paths.MODELS_DIR}")
    
    print("\n[Model]")
    print(f"Features: {settings.model.FEATURE_NAMES}")
    print(f"Available Models: {list(settings.model.AVAILABLE_MODELS.keys())}")
    
    print("\n[API]")
    print(f"Host: {settings.api.HOST}")
    print(f"Port: {settings.api.PORT}")
    
    print("\n[MLflow]")
    print(f"Tracking URI: {settings.mlflow.TRACKING_URI}")
    print(f"Experiment: {settings.mlflow.EXPERIMENT_NAME}")
    print(f"Enabled: {settings.mlflow.ENABLED}")