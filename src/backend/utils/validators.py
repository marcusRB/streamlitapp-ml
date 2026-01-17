"""
Data Validation Utilities
Validate patient data and predictions
"""

from typing import Dict, List, Tuple, Any
import pandas as pd
from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom validation error"""
    pass


class PatientDataValidator:
    """Validator for patient data"""
    
    def __init__(self):
        self.feature_names = settings.model.FEATURE_NAMES
        self.feature_ranges = settings.model.FEATURE_RANGES
    
    def validate_features(self, data: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate that all required features are present
        
        Args:
            data: Patient data dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check all required features are present
        missing_features = set(self.feature_names) - set(data.keys())
        if missing_features:
            errors.append(f"Missing required features: {', '.join(missing_features)}")
        
        # Check for extra features
        extra_features = set(data.keys()) - set(self.feature_names)
        if extra_features:
            logger.warning(f"Extra features will be ignored: {', '.join(extra_features)}")
        
        return len(errors) == 0, errors
    
    def validate_ranges(self, data: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate that feature values are within expected ranges
        
        Args:
            data: Patient data dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        for feature, value in data.items():
            if feature not in self.feature_ranges:
                continue
            
            min_val, max_val = self.feature_ranges[feature]
            
            if not (min_val <= value <= max_val):
                errors.append(
                    f"Feature '{feature}' value {value} is out of range "
                    f"[{min_val}, {max_val}]"
                )
        
        return len(errors) == 0, errors
    
    def validate_types(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that feature values are numeric
        
        Args:
            data: Patient data dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        for feature, value in data.items():
            if feature not in self.feature_names:
                continue
            
            try:
                float(value)
            except (TypeError, ValueError):
                errors.append(f"Feature '{feature}' has invalid type: {type(value)}")
        
        return len(errors) == 0, errors
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate patient data completely
        
        Args:
            data: Patient data dictionary
            
        Returns:
            Validation result dictionary
            
        Raises:
            ValidationError: If validation fails
        """
        all_errors = []
        
        # Validate features
        is_valid, errors = self.validate_features(data)
        if not is_valid:
            all_errors.extend(errors)
        
        # Validate types
        is_valid, errors = self.validate_types(data)
        if not is_valid:
            all_errors.extend(errors)
        
        # Validate ranges
        is_valid, errors = self.validate_ranges(data)
        if not is_valid:
            all_errors.extend(errors)
        
        if all_errors:
            raise ValidationError("; ".join(all_errors))
        
        return {
            'valid': True,
            'data': {k: float(v) for k, v in data.items() if k in self.feature_names},
            'message': 'Validation successful'
        }


class DataFrameValidator:
    """Validator for DataFrame data"""
    
    def __init__(self):
        self.feature_names = settings.model.FEATURE_NAMES
    
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame schema
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required columns
        missing_cols = set(self.feature_names) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Check for empty DataFrame
        if df.empty:
            errors.append("DataFrame is empty")
        
        return len(errors) == 0, errors
    
    def validate_data_types(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data types in DataFrame
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        for col in self.feature_names:
            if col not in df.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' should be numeric, got {df[col].dtype}")
        
        return len(errors) == 0, errors
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame completely
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation result dictionary
            
        Raises:
            ValidationError: If validation fails
        """
        all_errors = []
        
        # Validate schema
        is_valid, errors = self.validate_schema(df)
        if not is_valid:
            all_errors.extend(errors)
        
        # Validate data types
        is_valid, errors = self.validate_data_types(df)
        if not is_valid:
            all_errors.extend(errors)
        
        if all_errors:
            raise ValidationError("; ".join(all_errors))
        
        return {
            'valid': True,
            'shape': df.shape,
            'columns': list(df.columns),
            'message': 'Validation successful'
        }


class ModelValidator:
    """Validator for model-related operations"""
    
    @staticmethod
    def validate_model_name(model_name: str) -> bool:
        """
        Validate model name
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If model name is invalid
        """
        available_models = settings.model.AVAILABLE_MODELS.keys()
        
        if model_name not in available_models:
            raise ValidationError(
                f"Invalid model name '{model_name}'. "
                f"Available models: {', '.join(available_models)}"
            )
        
        return True
    
    @staticmethod
    def validate_prediction(prediction: int) -> bool:
        """
        Validate prediction value
        
        Args:
            prediction: Prediction value
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If prediction is invalid
        """
        if prediction not in [0, 1]:
            raise ValidationError(f"Invalid prediction value: {prediction}. Must be 0 or 1.")
        
        return True


def validate_patient_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to validate patient data
    
    Args:
        data: Patient data dictionary
        
    Returns:
        Validated data dictionary
    """
    validator = PatientDataValidator()
    return validator.validate(data)


def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function to validate DataFrame
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Validation result dictionary
    """
    validator = DataFrameValidator()
    return validator.validate(df)


def validate_model_name(model_name: str) -> bool:
    """
    Convenience function to validate model name
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if valid
    """
    return ModelValidator.validate_model_name(model_name)


if __name__ == "__main__":
    # Test validators
    print("Testing Patient Data Validator")
    print("=" * 60)
    
    # Valid data
    valid_data = {
        'hemo': 15.4, 'sg': 1.020, 'sc': 1.2,
        'rbcc': 5.2, 'pcv': 44.0, 'htn': 1.0,
        'dm': 1.0, 'bp': 80.0, 'age': 48.0
    }
    
    try:
        result = validate_patient_data(valid_data)
        print("✅ Valid data:", result)
    except ValidationError as e:
        print("❌ Error:", str(e))
    
    # Invalid data (missing features)
    print("\n" + "=" * 60)
    invalid_data = {'hemo': 15.4, 'sg': 1.020}
    
    try:
        result = validate_patient_data(invalid_data)
        print("✅ Valid data:", result)
    except ValidationError as e:
        print("❌ Error:", str(e))
    
    # Invalid data (out of range)
    print("\n" + "=" * 60)
    invalid_range_data = {
        'hemo': 25.0, 'sg': 1.020, 'sc': 1.2,
        'rbcc': 5.2, 'pcv': 44.0, 'htn': 1.0,
        'dm': 1.0, 'bp': 80.0, 'age': 48.0
    }
    
    try:
        result = validate_patient_data(invalid_range_data)
        print("✅ Valid data:", result)
    except ValidationError as e:
        print("❌ Error:", str(e))