# backend/api/routes/predictions.py
from fastapi import APIRouter, HTTPException
from backend.core.model_prediction import ModelPredictor
from backend.utils.config import settings
from backend.utils.validators import validate_patient_data

router = APIRouter()

@router.post("/")
async def predict(patient_data: dict, model_name: str = "KNN"):
    # Validate
    validated_data = validate_patient_data(patient_data)
    
    # Predict
    predictor = ModelPredictor(models_dir=settings.paths.MODELS_DIR)
    result = predictor.predict_single(model_name, validated_data['data'])
    
    return result