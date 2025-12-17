"""
ETS (Exponential Smoothing) forecasting service.

ETS models use weighted averages of past observations with exponentially
decreasing weights for older observations.
"""
import time
from typing import Dict, Any, Optional
import numpy as np
import warnings

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from services.base_analyzer import BaseAnalyzer
from models.forecast_result import ForecastResult
from models.time_series_data import TimeSeriesData
from utils.data_utils import DataUtils


class ETSService(BaseAnalyzer):
    """
    ETS (Error, Trend, Seasonal) forecasting service.
    
    Components:
    - Error: Additive or multiplicative
    - Trend: None, additive, or multiplicative
    - Seasonal: None, additive, or multiplicative
    
    Models include:
    - Simple Exponential Smoothing (no trend, no seasonality)
    - Holt's Linear: Additive trend
    - Holt-Winters: Trend + Seasonality
    
    Best for:
    - Data with trend and/or seasonality
    - Short to medium term forecasting
    - When recent observations are more important
    """
    
    def __init__(self):
        super().__init__("ETS")
        self.model = None
        self.fitted_model = None
        self.training_data = None
        self.trend = None
        self.seasonal = None
        self.seasonal_periods = None
    
    def fit(
        self,
        data: TimeSeriesData,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Fit ETS model on training data.
        
        Args:
            data: Training time series data
            trend: Trend component ('add', 'mul', or None)
            seasonal: Seasonal component ('add', 'mul', or None)
            seasonal_periods: Number of periods in a seasonal cycle
        """
        if not self.validate_data(data):
            raise ValueError("Invalid training data")
        
        self.training_data = data
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Convert to numpy array and handle edge cases
            values = np.array(data.values)
            
            # Ensure positive values for multiplicative models
            if trend == "mul" or seasonal == "mul":
                values = np.maximum(values, 0.01)
            
            self.model = ExponentialSmoothing(
                values,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods
            )
            self.fitted_model = self.model.fit(optimized=True)
    
    def predict(self, steps: int, **kwargs) -> ForecastResult:
        """
        Generate ETS predictions.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            ForecastResult with predictions
        """
        start_time = time.time()
        
        if self.fitted_model is None:
            return ForecastResult.failed(
                model_name="ETS",
                message="Model not fitted. Call fit() first."
            )
        
        try:
            # Generate forecasts
            forecast = self.fitted_model.forecast(steps)
            predictions = forecast.tolist()
            
            # Generate future timestamps
            if self.training_data and self.training_data.timestamps:
                last_timestamp = self.training_data.timestamps[-1]
                future_timestamps = DataUtils.generate_future_timestamps(
                    last_timestamp, steps
                )
            else:
                future_timestamps = [f"t+{i}" for i in range(1, steps + 1)]
            
            processing_time = time.time() - start_time
            
            return ForecastResult.success(
                model_name="ETS",
                message=f"Generated {steps} step forecast",
                predictions=predictions,
                prediction_timestamps=future_timestamps,
                model_params=self.get_model_params(),
                processing_time=processing_time
            )
            
        except Exception as e:
            return ForecastResult.failed(
                model_name="ETS",
                message=f"Forecasting failed: {str(e)}"
            )
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get ETS model parameters."""
        params = {
            "trend": self.trend,
            "seasonal": self.seasonal,
            "seasonal_periods": self.seasonal_periods
        }
        
        if self.fitted_model:
            params["smoothing_level"] = float(self.fitted_model.params.get("smoothing_level", 0))
            params["smoothing_trend"] = float(self.fitted_model.params.get("smoothing_trend", 0))
            params["aic"] = float(self.fitted_model.aic) if hasattr(self.fitted_model, "aic") else None
        
        return params
    
    def simple_exponential_smoothing(
        self,
        data: TimeSeriesData,
        steps: int
    ) -> ForecastResult:
        """Apply simple exponential smoothing (no trend, no seasonality)."""
        self.fit(data, trend=None, seasonal=None)
        return self.predict(steps)
    
    def holt_linear(
        self,
        data: TimeSeriesData,
        steps: int
    ) -> ForecastResult:
        """Apply Holt's linear method (additive trend, no seasonality)."""
        self.fit(data, trend="add", seasonal=None)
        return self.predict(steps)
    
    def holt_winters(
        self,
        data: TimeSeriesData,
        steps: int,
        seasonal_periods: int = 12
    ) -> ForecastResult:
        """Apply Holt-Winters method (additive trend + seasonality)."""
        self.fit(data, trend="add", seasonal="add", seasonal_periods=seasonal_periods)
        return self.predict(steps)
