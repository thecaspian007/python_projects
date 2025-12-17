"""
Abstract base class for forecasting analyzers.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np

from models.forecast_result import ForecastResult
from models.time_series_data import TimeSeriesData


class BaseAnalyzer(ABC):
    """Abstract base class for time series analysis services."""
    
    def __init__(self, name: str):
        """Initialize analyzer with a name."""
        self.name = name
    
    @abstractmethod
    def fit(self, data: TimeSeriesData, **kwargs) -> None:
        """
        Fit the model on training data.
        
        Args:
            data: Training time series data
            **kwargs: Additional model parameters
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int, **kwargs) -> ForecastResult:
        """
        Generate predictions.
        
        Args:
            steps: Number of steps to forecast
            **kwargs: Additional parameters
            
        Returns:
            ForecastResult with predictions
        """
        pass
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get current model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        pass
    
    def validate_data(self, data: TimeSeriesData) -> bool:
        """Validate that the data is valid for analysis."""
        if data is None:
            return False
        if data.length < 2:
            return False
        return True
    
    def fit_predict(
        self,
        data: TimeSeriesData,
        forecast_steps: int,
        **kwargs
    ) -> ForecastResult:
        """
        Fit model and generate predictions in one step.
        
        Args:
            data: Training data
            forecast_steps: Number of steps to forecast
            **kwargs: Additional parameters
            
        Returns:
            ForecastResult with predictions
        """
        self.fit(data, **kwargs)
        return self.predict(forecast_steps, **kwargs)
