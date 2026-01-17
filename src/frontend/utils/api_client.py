"""
API Client for Frontend
Handles communication with backend FastAPI server
"""

import requests
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class BackendAPIClient:
    """Client for communicating with backend API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of the backend API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def _handle_response(self, response: requests.Response) -> Dict:
        """
        Handle API response
        
        Args:
            response: Response object
            
        Returns:
            Response JSON data
            
        Raises:
            Exception: If request fails
        """
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise
    
    def health_check(self) -> Dict:
        """
        Check API health
        
        Returns:
            Health status dictionary
        """
        try:
            response = self.session.get(f"{self.base_url}/health")
            return self._handle_response(response)
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def list_models(self) -> Dict:
        """
        List available models
        
        Returns:
            Models dictionary
        """
        response = self.session.get(f"{self.base_url}/models")
        return self._handle_response(response)
    
    def predict(
        self, 
        patient_data: Dict, 
        model_name: str = "KNN"
    ) -> Dict:
        """
        Make prediction for single patient
        
        Args:
            patient_data: Patient data dictionary
            model_name: Name of model to use
            
        Returns:
            Prediction response
        """
        response = self.session.post(
            f"{self.base_url}/predict",
            params={"model_name": model_name},
            json=patient_data
        )
        return self._handle_response(response)
    
    def predict_batch(
        self, 
        patients: List[Dict], 
        model_name: str = "KNN"
    ) -> Dict:
        """
        Make predictions for multiple patients
        
        Args:
            patients: List of patient data dictionaries
            model_name: Name of model to use
            
        Returns:
            Batch prediction response
        """
        response = self.session.post(
            f"{self.base_url}/predict/batch",
            params={"model_name": model_name},
            json={"patients": patients}
        )
        return self._handle_response(response)
    
    def predict_ensemble(self, patient_data: Dict) -> Dict:
        """
        Make prediction using all models
        
        Args:
            patient_data: Patient data dictionary
            
        Returns:
            Ensemble prediction response
        """
        response = self.session.post(
            f"{self.base_url}/predict/ensemble",
            json=patient_data
        )
        return self._handle_response(response)
    
    def get_mlflow_experiments(self) -> Dict:
        """
        Get MLflow experiments
        
        Returns:
            Experiments dictionary
        """
        try:
            response = self.session.get(f"{self.base_url}/mlflow/experiments")
            return self._handle_response(response)
        except Exception as e:
            logger.warning(f"MLflow not available: {str(e)}")
            return {"experiments": [], "total": 0}
    
    def get_mlflow_runs(
        self, 
        experiment_name: str, 
        limit: int = 10
    ) -> Dict:
        """
        Get MLflow runs for an experiment
        
        Args:
            experiment_name: Name of the experiment
            limit: Maximum number of runs to return
            
        Returns:
            Runs dictionary
        """
        try:
            response = self.session.get(
                f"{self.base_url}/mlflow/runs/{experiment_name}",
                params={"limit": limit}
            )
            return self._handle_response(response)
        except Exception as e:
            logger.warning(f"Error fetching runs: {str(e)}")
            return {"runs": [], "total": 0}
    
    def is_connected(self) -> bool:
        """
        Check if backend is reachable
        
        Returns:
            True if connected, False otherwise
        """
        try:
            health = self.health_check()
            return health.get("status") == "healthy"
        except Exception:
            return False


# Singleton instance
_client = None


def get_api_client(base_url: str = "http://localhost:8000") -> BackendAPIClient:
    """
    Get or create API client instance
    
    Args:
        base_url: Base URL of the backend API
        
    Returns:
        BackendAPIClient instance
    """
    global _client
    
    if _client is None:
        _client = BackendAPIClient(base_url)
    
    return _client


if __name__ == "__main__":
    # Test API client
    client = get_api_client()
    
    print("Testing Backend API Client")
    print("=" * 60)
    
    # Health check
    print("\n[1] Health Check")
    health = client.health_check()
    print(f"Status: {health.get('status')}")
    print(f"Connected: {client.is_connected()}")
    
    if client.is_connected():
        # List models
        print("\n[2] List Models")
        models = client.list_models()
        print(f"Available models: {list(models.get('models', {}).keys())}")
        
        # Test prediction
        print("\n[3] Test Prediction")
        patient = {
            'hemo': 15.4, 'sg': 1.020, 'sc': 1.2,
            'rbcc': 5.2, 'pcv': 44.0, 'htn': 1.0,
            'dm': 1.0, 'bp': 80.0, 'age': 48.0
        }
        
        result = client.predict(patient, model_name="KNN")
        print(f"Prediction: {result.get('prediction')}")
        print(f"Confidence: {result.get('confidence', 0):.2%}")
    else:
        print("\n❌ Backend API is not running!")
        print("Start the backend with: python backend/api/main.py")