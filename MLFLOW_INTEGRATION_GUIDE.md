# MLflow Integration Guide for CKD Detection Project

## Overview

MLflow will help you:
- **Track experiments** (parameters, metrics, models)
- **Compare model performance** across runs
- **Version models** with automatic logging
- **Serve models** via REST API
- **Visualize** training history in web UI

## Installation

Add to `requirements.txt`:
```bash
mlflow==2.9.2
```

Install:
```bash
pip install mlflow==2.9.2
```

## Project Structure Updates

```
ckd-detection/
├── mlruns/                    # MLflow tracking data (auto-created)
├── mlflow_artifacts/          # Model artifacts
├── src/
│   ├── model_training.py      # Updated with MLflow
│   ├── model_prediction.py    # Updated with MLflow
│   └── mlflow_config.py       # MLflow configuration (NEW)
└── mlflow.db                  # SQLite backend (optional)
```

## Implementation

### 1. MLflow Configuration Module

Create `src/mlflow_config.py`:

```python
"""
MLflow Configuration for CKD Detection Project
"""

import mlflow
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MLflowConfig:
    """Central MLflow configuration"""
    
    def __init__(self, experiment_name: str = "CKD_Detection"):
        """
        Initialize MLflow configuration
        
        Args:
            experiment_name: Name of MLflow experiment
        """
        self.experiment_name = experiment_name
        self.tracking_uri = Path("../../mlruns").resolve()
        self.setup()
    
    def setup(self):
        """Setup MLflow tracking"""
        # Set tracking URI
        mlflow.set_tracking_uri(f"file://{self.tracking_uri}")
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=str(self.tracking_uri / "artifacts")
                )
                logger.info(f"Created experiment: {self.experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error setting up MLflow: {str(e)}")
            raise
    
    @staticmethod
    def start_run(run_name: str = None):
        """Start MLflow run"""
        return mlflow.start_run(run_name=run_name)
    
    @staticmethod
    def end_run():
        """End MLflow run"""
        mlflow.end_run()
    
    @staticmethod
    def log_params(params: dict):
        """Log parameters"""
        mlflow.log_params(params)
    
    @staticmethod
    def log_metrics(metrics: dict):
        """Log metrics"""
        mlflow.log_metrics(metrics)
    
    @staticmethod
    def log_model(model, artifact_path: str):
        """Log model"""
        mlflow.sklearn.log_model(model, artifact_path)
    
    @staticmethod
    def log_artifact(file_path: str):
        """Log artifact file"""
        mlflow.log_artifact(file_path)
```

### 2. Updated model_training.py with MLflow

Key changes to integrate:

```python
# Add imports
import mlflow
import mlflow.sklearn
from mlflow_config import MLflowConfig

class ModelTrainer:
    def __init__(self):
        # ... existing code ...
        
        # Initialize MLflow
        self.mlflow_config = MLflowConfig(experiment_name="CKD_Detection")
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    
    def train_knn(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train KNN model with MLflow tracking"""
        logger.info("Training KNN model...")
        
        # Start MLflow run
        with mlflow.start_run(run_name="KNN_Training"):
            # Log dataset info
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            
            # Define model and parameter grid
            knn = KNeighborsClassifier()
            param_grid = {'n_neighbors': np.arange(1, 31, 2)}
            
            # Log search space
            mlflow.log_param("search_space", "n_neighbors: 1-30 (step 2)")
            
            # GridSearch
            grid_search = GridSearchCV(knn, param_grid, cv=10, scoring='f1', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            # Best parameters
            best_k = grid_search.best_params_['n_neighbors']
            best_score = grid_search.best_score_
            
            # Log hyperparameters
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("cv_f1_score", best_score)
            
            # Train final model
            knn_final = KNeighborsClassifier(n_neighbors=best_k)
            knn_final.fit(X_train, y_train)
            
            # Predictions
            y_pred = knn_final.predict(X_test)
            
            # Calculate and log metrics
            metrics = {
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_precision': precision_score(y_test, y_pred),
                'test_recall': recall_score(y_test, y_pred),
                'test_f1_score': f1_score(y_test, y_pred)
            }
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                knn_final, 
                "model",
                registered_model_name="KNN_CKD_Detector"
            )
            
            # Log confusion matrix plot
            cm_path = self.figures_dir / 'knn_confusion_matrix.png'
            if cm_path.exists():
                mlflow.log_artifact(str(cm_path))
            
            # Log optimization plot
            opt_path = self.figures_dir / 'knn_optimization.png'
            if opt_path.exists():
                mlflow.log_artifact(str(opt_path))
            
            # Save run ID for later reference
            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
            
            # ... rest of existing code ...
            
            results['mlflow_run_id'] = run_id
            return results
```

### 3. Updated model_prediction.py with MLflow

```python
# Add imports
import mlflow
import mlflow.sklearn
from mlflow_config import MLflowConfig

class ModelPredictor:
    def __init__(self, models_dir: str = '../../models', use_mlflow: bool = True):
        # ... existing code ...
        
        self.use_mlflow = use_mlflow
        if use_mlflow:
            self.mlflow_config = MLflowConfig()
    
    def load_model_from_mlflow(self, model_name: str, version: str = "latest"):
        """
        Load model from MLflow Model Registry
        
        Args:
            model_name: Registered model name
            version: Model version or "latest"
        """
        try:
            if version == "latest":
                model_uri = f"models:/{model_name}/latest"
            else:
                model_uri = f"models:/{model_name}/{version}"
            
            model = mlflow.sklearn.load_model(model_uri)
            self.models[model_name] = model
            logger.info(f"Loaded {model_name} (version: {version}) from MLflow")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {str(e)}")
            raise
    
    def load_model_by_run_id(self, run_id: str, model_name: str):
        """
        Load model from specific MLflow run
        
        Args:
            run_id: MLflow run ID
            model_name: Name to assign to loaded model
        """
        try:
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            self.models[model_name] = model
            logger.info(f"Loaded model from run {run_id}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model by run ID: {str(e)}")
            raise
    
    def predict_single(self, model_name: str, features: Dict) -> Dict:
        """Make prediction with MLflow logging"""
        
        if self.use_mlflow:
            with mlflow.start_run(run_name=f"Prediction_{model_name}"):
                # Log input features
                mlflow.log_params({f"input_{k}": v for k, v in features.items()})
                
                # Make prediction (existing code)
                result = self._predict_single_internal(model_name, features)
                
                # Log prediction
                mlflow.log_metric("prediction", result['prediction_numeric'])
                if result['probability']:
                    mlflow.log_metric("confidence", 
                                    result['probability'][result['prediction']])
                
                # Log run ID
                result['mlflow_run_id'] = mlflow.active_run().info.run_id
                
                return result
        else:
            return self._predict_single_internal(model_name, features)
    
    def _predict_single_internal(self, model_name: str, features: Dict) -> Dict:
        """Internal prediction logic (existing code)"""
        # ... your existing predict_single code ...
```

### 4. Integration in Streamlit App

Update `app.py`:

```python
import mlflow
from mlflow_config import MLflowConfig

def init_session_state():
    # ... existing code ...
    
    # Initialize MLflow
    if 'mlflow_config' not in st.session_state:
        st.session_state.mlflow_config = MLflowConfig()

def model_training_section():
    """Model training section with MLflow"""
    st.header("🤖 Model Training")
    
    # Add MLflow UI link
    st.info("🔗 View experiments in MLflow UI: Run `mlflow ui` in terminal")
    
    # ... existing training code ...
    
    # After training, show MLflow run IDs
    if results:
        st.subheader("MLflow Run IDs")
        for model_name, result in results.items():
            if 'mlflow_run_id' in result:
                st.code(f"{model_name}: {result['mlflow_run_id']}")

def mlflow_ui_section():
    """New section to display MLflow experiments"""
    st.header("📊 MLflow Experiments")
    
    try:
        # Get all experiments
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        
        for exp in experiments:
            with st.expander(f"Experiment: {exp.name}"):
                runs = client.search_runs(exp.experiment_id)
                
                if runs:
                    # Create dataframe from runs
                    runs_data = []
                    for run in runs:
                        runs_data.append({
                            'Run Name': run.data.tags.get('mlflow.runName', 'N/A'),
                            'Status': run.info.status,
                            'F1 Score': run.data.metrics.get('test_f1_score', 'N/A'),
                            'Accuracy': run.data.metrics.get('test_accuracy', 'N/A'),
                            'Start Time': run.info.start_time,
                            'Run ID': run.info.run_id
                        })
                    
                    df_runs = pd.DataFrame(runs_data)
                    st.dataframe(df_runs, use_container_width=True)
                else:
                    st.info("No runs found")
    
    except Exception as e:
        st.error(f"Error loading MLflow data: {str(e)}")
```

## Usage

### 1. Start MLflow UI

```bash
# From project root
mlflow ui --backend-store-uri file:///absolute/path/to/mlruns

# Or simply (if in project directory)
mlflow ui
```

Access at: `http://localhost:5000`

### 2. Train Models with MLflow

```bash
cd src
python model_training.py
```

All runs will be logged automatically!

### 3. View Experiments

Open browser: `http://localhost:5000`

You'll see:
- All experiment runs
- Parameters (k, C, gamma, etc.)
- Metrics (accuracy, F1, precision, recall)
- Artifacts (models, plots)
- Model registry

### 4. Load Model from MLflow

```python
from model_prediction import ModelPredictor

predictor = ModelPredictor(use_mlflow=True)

# Load latest registered model
predictor.load_model_from_mlflow("KNN_CKD_Detector", version="latest")

# Or load from specific run
predictor.load_model_by_run_id("abc123def456", "KNN")
```

### 5. Compare Models

In MLflow UI:
1. Select multiple runs
2. Click "Compare"
3. View side-by-side metrics
4. Analyze parameter impact

## MLflow Features Used

### 1. Experiment Tracking
- Automatic logging of parameters
- Metric tracking across runs
- Artifact storage (plots, models)

### 2. Model Registry
- Version control for models
- Stage management (Staging, Production)
- Model lineage tracking

### 3. Model Serving (Optional)

```bash
# Serve model via REST API
mlflow models serve -m "models:/KNN_CKD_Detector/latest" -p 5001

# Make prediction
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": ["hemo", "sg", "sc", ...], "data": [[15.4, 1.020, ...]]}}'
```

## Best Practices

### 1. Organized Runs
```python
with mlflow.start_run(run_name=f"KNN_k{best_k}_{timestamp}"):
    # training code
```

### 2. Tag Runs
```python
mlflow.set_tag("model_type", "KNN")
mlflow.set_tag("dataset_version", "v1.0")
mlflow.set_tag("experiment_phase", "hyperparameter_tuning")
```

### 3. Log Everything Important
```python
mlflow.log_param("preprocessing", "normalized")
mlflow.log_metric("training_time_seconds", training_time)
mlflow.log_artifact("feature_importance.csv")
```

### 4. Model Registry Workflow
```python
# Register model
mlflow.register_model("runs:/abc123/model", "KNN_CKD_Detector")

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="KNN_CKD_Detector",
    version=1,
    stage="Production"
)
```

## Advanced: Autolog

For automatic logging:

```python
# In model_training.py
mlflow.sklearn.autolog()

# Now all sklearn models log automatically!
knn_final.fit(X_train, y_train)
# Parameters, metrics, and model logged automatically
```

## Troubleshooting

### Issue: MLflow UI not showing data
```bash
# Check tracking URI
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Ensure mlruns directory exists
ls -la mlruns/
```

### Issue: Can't load model
```bash
# List all registered models
mlflow models list

# Check model versions
mlflow models get-model-version --name "KNN_CKD_Detector" --version 1
```

## Summary

MLflow integration provides:
- ✅ **Experiment tracking** - All runs logged automatically
- ✅ **Model versioning** - Track model evolution
- ✅ **Comparison** - Compare runs side-by-side
- ✅ **Reproducibility** - Recreate any run
- ✅ **Deployment** - Easy model serving
- ✅ **Collaboration** - Share results with team