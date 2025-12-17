"""
Moving Average forecasting service.

Moving averages are simple yet effective methods for smoothing time series
data and making short-term forecasts.
"""
import time
from typing import Dict, Any, List, Optional
import numpy as np

from services.base_analyzer import BaseAnalyzer
from models.forecast_result import ForecastResult
from models.time_series_data import TimeSeriesData
from utils.data_utils import DataUtils


class MovingAverageService(BaseAnalyzer):
    """
    Moving Average forecasting service.
    
    Types:
    - Simple Moving Average (SMA): Equal weights for all periods
    - Weighted Moving Average (WMA): Higher weights for recent periods
    - Exponential Moving Average (EMA): Exponentially decreasing weights
    
    Best for:
    - Short-term forecasting
    - Trend identification
    - Data smoothing
    """
    
    def __init__(self):
        super().__init__("MovingAverage")
        self.training_data = None
        self.window_size = 5
        self.method = "sma"
    
    def fit(
        self,
        data: TimeSeriesData,
        window_size: int = 5,
        method: str = "sma",
        **kwargs
    ) -> None:
        """
        Store training data and parameters.
        
        Args:
            data: Training time series data
            window_size: Number of periods for moving average
            method: Type of moving average (sma, wma, ema)
        """
        if not self.validate_data(data):
            raise ValueError("Invalid training data")
        
        self.training_data = data
        self.window_size = min(window_size, data.length)
        self.method = method
    
    def predict(self, steps: int, **kwargs) -> ForecastResult:
        """
        Generate moving average predictions.
        
        For forecasting, moving averages project the last calculated
        average value forward.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            ForecastResult with predictions
        """
        start_time = time.time()
        
        if self.training_data is None:
            return ForecastResult.failed(
                model_name="MovingAverage",
                message="Model not fitted. Call fit() first."
            )
        
        try:
            values = np.array(self.training_data.values)
            
            # Calculate moving average based on method
            if self.method == "sma":
                ma_values = self._simple_moving_average(values)
            elif self.method == "wma":
                ma_values = self._weighted_moving_average(values)
            elif self.method == "ema":
                ma_values = self._exponential_moving_average(values)
            else:
                return ForecastResult.failed(
                    model_name="MovingAverage",
                    message=f"Unknown method: {self.method}"
                )
            
            # For forecasting, use the last MA value and trend
            last_ma = ma_values[-1]
            
            # Calculate trend from last few MA values
            if len(ma_values) >= 2:
                trend = ma_values[-1] - ma_values[-2]
            else:
                trend = 0
            
            # Generate predictions with trend
            predictions = []
            for i in range(1, steps + 1):
                pred = last_ma + (trend * i)
                predictions.append(float(pred))
            
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
                model_name=f"MovingAverage_{self.method.upper()}",
                message=f"Generated {steps} step forecast using {self.method.upper()}",
                predictions=predictions,
                prediction_timestamps=future_timestamps,
                model_params=self.get_model_params(),
                processing_time=processing_time
            )
            
        except Exception as e:
            return ForecastResult.failed(
                model_name="MovingAverage",
                message=f"Forecasting failed: {str(e)}"
            )
    
    def _simple_moving_average(self, values: np.ndarray) -> np.ndarray:
        """Calculate Simple Moving Average."""
        weights = np.ones(self.window_size) / self.window_size
        return np.convolve(values, weights, mode='valid')
    
    def _weighted_moving_average(self, values: np.ndarray) -> np.ndarray:
        """Calculate Weighted Moving Average (linear weights)."""
        weights = np.arange(1, self.window_size + 1, dtype=float)
        weights = weights / weights.sum()
        return np.convolve(values, weights[::-1], mode='valid')
    
    def _exponential_moving_average(
        self,
        values: np.ndarray,
        alpha: Optional[float] = None
    ) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if alpha is None:
            alpha = 2 / (self.window_size + 1)
        
        ema = np.zeros(len(values))
        ema[0] = values[0]
        
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get moving average parameters."""
        return {
            "method": self.method,
            "window_size": self.window_size
        }
    
    def smooth_series(self, data: TimeSeriesData) -> List[float]:
        """
        Apply moving average to smooth a time series.
        
        Args:
            data: Time series data to smooth
            
        Returns:
            List of smoothed values
        """
        self.fit(data)
        values = np.array(data.values)
        
        if self.method == "sma":
            smoothed = self._simple_moving_average(values)
        elif self.method == "wma":
            smoothed = self._weighted_moving_average(values)
        else:
            smoothed = self._exponential_moving_average(values)
        
        return smoothed.tolist()
