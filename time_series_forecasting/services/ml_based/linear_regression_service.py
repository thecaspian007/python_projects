"""
Linear Regression forecasting service.

Linear regression models the relationship between time and target variable
using a linear function.
"""
import time
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from services.base_analyzer import BaseAnalyzer
from models.forecast_result import ForecastResult
from models.time_series_data import TimeSeriesData
from utils.data_utils import DataUtils


class LinearRegressionService(BaseAnalyzer):
    """
    Linear Regression forecasting service.
    
    Features:
    - Simple linear regression (trend only)
    - Polynomial regression for non-linear trends
    - Multiple feature support
    
    Best for:
    - Data with clear linear or polynomial trends
    - When simplicity and interpretability are important
    - Baseline forecasting model
    """
    
    def __init__(self):
        super().__init__("LinearRegression")
        self.model = LinearRegression()
        self.poly_features = None
        self.training_data = None
        self.degree = 1
    
    def fit(
        self,
        data: TimeSeriesData,
        degree: int = 1,
        **kwargs
    ) -> None:
        """
        Fit linear regression model on training data.
        
        Args:
            data: Training time series data
            degree: Polynomial degree (1 = linear)
        """
        if not self.validate_data(data):
            raise ValueError("Invalid training data")
        
        self.training_data = data
        self.degree = degree
        
        # Create time index as feature
        X = np.arange(len(data.values)).reshape(-1, 1)
        y = np.array(data.values)
        
        # Apply polynomial features if degree > 1
        if degree > 1:
            self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X = self.poly_features.fit_transform(X)
        
        self.model.fit(X, y)
    
    def predict(self, steps: int, **kwargs) -> ForecastResult:
        """
        Generate linear regression predictions.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            ForecastResult with predictions
        """
        start_time = time.time()
        
        if self.training_data is None:
            return ForecastResult.failed(
                model_name="LinearRegression",
                message="Model not fitted. Call fit() first."
            )
        
        try:
            # Create future time indices
            last_idx = len(self.training_data.values)
            future_indices = np.arange(last_idx, last_idx + steps).reshape(-1, 1)
            
            # Apply same polynomial transformation
            if self.poly_features:
                future_indices = self.poly_features.transform(future_indices)
            
            # Predict
            predictions = self.model.predict(future_indices).tolist()
            
            # Generate future timestamps
            if self.training_data.timestamps:
                last_timestamp = self.training_data.timestamps[-1]
                future_timestamps = DataUtils.generate_future_timestamps(
                    last_timestamp, steps
                )
            else:
                future_timestamps = [f"t+{i}" for i in range(1, steps + 1)]
            
            processing_time = time.time() - start_time
            
            return ForecastResult.success(
                model_name=f"LinearRegression_degree{self.degree}",
                message=f"Generated {steps} step forecast",
                predictions=predictions,
                prediction_timestamps=future_timestamps,
                model_params=self.get_model_params(),
                processing_time=processing_time
            )
            
        except Exception as e:
            return ForecastResult.failed(
                model_name="LinearRegression",
                message=f"Forecasting failed: {str(e)}"
            )
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get linear regression parameters."""
        params = {
            "degree": self.degree,
            "coefficients": self.model.coef_.tolist() if hasattr(self.model, "coef_") else None,
            "intercept": float(self.model.intercept_) if hasattr(self.model, "intercept_") else None
        }
        return params
    
    def get_fitted_values(self) -> List[float]:
        """Get fitted values for training data."""
        if self.training_data is None:
            return []
        
        X = np.arange(len(self.training_data.values)).reshape(-1, 1)
        
        if self.poly_features:
            X = self.poly_features.transform(X)
        
        return self.model.predict(X).tolist()
    
    def calculate_r_squared(self) -> float:
        """Calculate R-squared for the model."""
        if self.training_data is None:
            return 0.0
        
        fitted = self.get_fitted_values()
        actual = self.training_data.values
        
        ss_res = sum((a - f) ** 2 for a, f in zip(actual, fitted))
        ss_tot = sum((a - np.mean(actual)) ** 2 for a in actual)
        
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
