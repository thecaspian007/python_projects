"""
Metrics service for evaluating forecast accuracy.

Provides various error metrics for comparing predicted vs actual values.
"""
from typing import Dict, List, Optional
import numpy as np

from models.forecast_result import EvaluationResult


class MetricsService:
    """
    Service for calculating forecasting evaluation metrics.
    
    Metrics provided:
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error
    - MSE: Mean Squared Error
    - R²: Coefficient of Determination
    - SMAPE: Symmetric Mean Absolute Percentage Error
    """
    
    @staticmethod
    def calculate_rmse(actual: List[float], predicted: List[float]) -> float:
        """
        Calculate Root Mean Squared Error.
        
        RMSE = sqrt(mean((actual - predicted)^2))
        
        Interpretation:
        - Lower is better
        - Same units as the target variable
        - Penalizes large errors more than MAE
        """
        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)
        return float(np.sqrt(np.mean((actual_arr - predicted_arr) ** 2)))
    
    @staticmethod
    def calculate_mae(actual: List[float], predicted: List[float]) -> float:
        """
        Calculate Mean Absolute Error.
        
        MAE = mean(|actual - predicted|)
        
        Interpretation:
        - Lower is better
        - Same units as target variable
        - More robust to outliers than RMSE
        """
        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)
        return float(np.mean(np.abs(actual_arr - predicted_arr)))
    
    @staticmethod
    def calculate_mse(actual: List[float], predicted: List[float]) -> float:
        """
        Calculate Mean Squared Error.
        
        MSE = mean((actual - predicted)^2)
        """
        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)
        return float(np.mean((actual_arr - predicted_arr) ** 2))
    
    @staticmethod
    def calculate_mape(actual: List[float], predicted: List[float]) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        MAPE = mean(|actual - predicted| / |actual|) * 100
        
        Interpretation:
        - Lower is better
        - Expressed as percentage
        - Undefined when actual values are zero
        """
        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)
        
        # Avoid division by zero
        mask = actual_arr != 0
        if not np.any(mask):
            return float('inf')
        
        return float(np.mean(np.abs((actual_arr[mask] - predicted_arr[mask]) / actual_arr[mask])) * 100)
    
    @staticmethod
    def calculate_smape(actual: List[float], predicted: List[float]) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.
        
        SMAPE = mean(2 * |actual - predicted| / (|actual| + |predicted|)) * 100
        
        Interpretation:
        - Lower is better
        - Bounded between 0 and 200%
        - Handles zero values better than MAPE
        """
        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)
        
        denominator = np.abs(actual_arr) + np.abs(predicted_arr)
        mask = denominator != 0
        
        if not np.any(mask):
            return 0.0
        
        return float(
            np.mean(
                2 * np.abs(actual_arr[mask] - predicted_arr[mask]) / denominator[mask]
            ) * 100
        )
    
    @staticmethod
    def calculate_r_squared(actual: List[float], predicted: List[float]) -> float:
        """
        Calculate R-squared (Coefficient of Determination).
        
        R² = 1 - (SS_res / SS_tot)
        
        Interpretation:
        - 1.0 = Perfect predictions
        - 0.0 = Model predicts as well as the mean
        - < 0 = Model is worse than predicting the mean
        """
        actual_arr = np.array(actual)
        predicted_arr = np.array(predicted)
        
        ss_res = np.sum((actual_arr - predicted_arr) ** 2)
        ss_tot = np.sum((actual_arr - np.mean(actual_arr)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return float(1 - (ss_res / ss_tot))
    
    @classmethod
    def calculate_all_metrics(
        cls,
        actual: List[float],
        predicted: List[float]
    ) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            actual: List of actual values
            predicted: List of predicted values
            
        Returns:
            Dictionary with all metrics
        """
        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted must have the same length")
        
        return {
            "rmse": cls.calculate_rmse(actual, predicted),
            "mae": cls.calculate_mae(actual, predicted),
            "mse": cls.calculate_mse(actual, predicted),
            "mape": cls.calculate_mape(actual, predicted),
            "smape": cls.calculate_smape(actual, predicted),
            "r_squared": cls.calculate_r_squared(actual, predicted)
        }
    
    @classmethod
    def evaluate(
        cls,
        model_name: str,
        actual: List[float],
        predicted: List[float]
    ) -> EvaluationResult:
        """
        Create full evaluation result.
        
        Args:
            model_name: Name of the model being evaluated
            actual: List of actual values
            predicted: List of predicted values
            
        Returns:
            EvaluationResult with all metrics
        """
        metrics = cls.calculate_all_metrics(actual, predicted)
        
        return EvaluationResult(
            model_name=model_name,
            metrics=metrics,
            predictions=predicted,
            actual_values=actual
        )
    
    @classmethod
    def compare_models(
        cls,
        actual: List[float],
        model_predictions: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models on the same data.
        
        Args:
            actual: List of actual values
            model_predictions: Dictionary mapping model names to predictions
            
        Returns:
            Dictionary mapping model names to their metrics
        """
        comparisons = {}
        
        for model_name, predictions in model_predictions.items():
            comparisons[model_name] = cls.calculate_all_metrics(actual, predictions)
        
        return comparisons
    
    @staticmethod
    def get_metric_descriptions() -> Dict[str, str]:
        """Get descriptions of all available metrics."""
        return {
            "rmse": "Root Mean Squared Error - Penalizes large errors",
            "mae": "Mean Absolute Error - Average absolute deviation",
            "mse": "Mean Squared Error - Average squared deviation",
            "mape": "Mean Absolute Percentage Error - Percentage error",
            "smape": "Symmetric MAPE - Handles zeros better than MAPE",
            "r_squared": "Coefficient of Determination - Explained variance"
        }
