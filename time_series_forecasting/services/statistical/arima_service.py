"""
ARIMA forecasting service.

ARIMA (AutoRegressive Integrated Moving Average) is a popular statistical
method for time series forecasting that combines autoregression, differencing,
and moving average components.
"""
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from services.base_analyzer import BaseAnalyzer
from models.forecast_result import ForecastResult
from models.time_series_data import TimeSeriesData
from utils.data_utils import DataUtils
from config.settings import settings


class ARIMAService(BaseAnalyzer):
    """
    ARIMA forecasting service.
    
    ARIMA(p, d, q) Model Components:
    - p: Number of autoregressive terms
    - d: Number of differences needed for stationarity
    - q: Number of moving average terms
    
    Best for:
    - Univariate time series
    - Data with trend and/or autocorrelation
    - Short to medium term forecasting
    """
    
    def __init__(self):
        super().__init__("ARIMA")
        self.model = None
        self.fitted_model = None
        self.order = settings.ARIMA_DEFAULT_ORDER
        self.training_data = None
    
    def fit(
        self,
        data: TimeSeriesData,
        order: Optional[Tuple[int, int, int]] = None,
        auto_order: bool = False,
        **kwargs
    ) -> None:
        """
        Fit ARIMA model on training data.
        
        Args:
            data: Training time series data
            order: ARIMA order (p, d, q)
            auto_order: Automatically determine order
        """
        if not self.validate_data(data):
            raise ValueError("Invalid training data")
        
        self.training_data = data
        
        if order:
            self.order = order
        elif auto_order:
            self.order = self._determine_order(data.values)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = ARIMA(data.values, order=self.order)
            self.fitted_model = self.model.fit()
    
    def predict(self, steps: int, **kwargs) -> ForecastResult:
        """
        Generate ARIMA predictions.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            ForecastResult with predictions
        """
        start_time = time.time()
        
        if self.fitted_model is None:
            return ForecastResult.failed(
                model_name="ARIMA",
                message="Model not fitted. Call fit() first."
            )
        
        try:
            # Generate forecasts
            forecast = self.fitted_model.forecast(steps=steps)
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
                model_name="ARIMA",
                message=f"Generated {steps} step forecast",
                predictions=predictions,
                prediction_timestamps=future_timestamps,
                model_params=self.get_model_params(),
                processing_time=processing_time
            )
            
        except Exception as e:
            return ForecastResult.failed(
                model_name="ARIMA",
                message=f"Forecasting failed: {str(e)}"
            )
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get ARIMA model parameters."""
        params = {
            "order": self.order,
            "p": self.order[0],
            "d": self.order[1],
            "q": self.order[2]
        }
        
        if self.fitted_model:
            params["aic"] = float(self.fitted_model.aic)
            params["bic"] = float(self.fitted_model.bic)
        
        return params
    
    def _determine_order(self, data: List[float]) -> Tuple[int, int, int]:
        """
        Automatically determine ARIMA order using statistical tests.
        
        This is a simplified auto-ARIMA implementation.
        """
        arr = np.array(data)
        
        # Test for stationarity using ADF test
        d = 0
        current_data = arr.copy()
        
        for i in range(3):
            try:
                adf_result = adfuller(current_data)
                if adf_result[1] < 0.05:  # p-value < 0.05 means stationary
                    break
                d += 1
                current_data = np.diff(current_data)
            except:
                break
        
        # Simple heuristic for p and q
        p = min(5, len(data) // 10)
        q = min(2, p)
        
        return (p, d, q)
    
    def get_model_summary(self) -> Optional[str]:
        """Get fitted model summary."""
        if self.fitted_model:
            return str(self.fitted_model.summary())
        return None
