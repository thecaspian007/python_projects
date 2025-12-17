"""
Neural Network forecasting service.

Uses a simple feedforward neural network for time series forecasting.
"""
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import warnings

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from services.base_analyzer import BaseAnalyzer
from models.forecast_result import ForecastResult
from models.time_series_data import TimeSeriesData
from utils.data_utils import DataUtils


class NeuralNetworkService(BaseAnalyzer):
    """
    Neural Network forecasting service.
    
    Features:
    - Configurable architecture
    - Sequence-based input for temporal patterns
    - Support for different activation functions
    
    Best for:
    - Complex non-linear patterns
    - Large datasets
    - When other methods underperform
    """
    
    def __init__(self):
        super().__init__("NeuralNetwork")
        self.model = None
        self.training_data = None
        self.sequence_length = 10
        self.min_val = 0
        self.max_val = 1
        self.epochs = 50
        self.hidden_units = [50, 25]
    
    def _create_model(self, input_shape: Tuple[int, ...], hidden_units: List[int]):
        """Create neural network model using Keras."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            
            model = Sequential()
            
            # Input layer
            model.add(Dense(hidden_units[0], activation='relu', input_shape=input_shape))
            model.add(Dropout(0.2))
            
            # Hidden layers
            for units in hidden_units[1:]:
                model.add(Dense(units, activation='relu'))
                model.add(Dropout(0.2))
            
            # Output layer
            model.add(Dense(1))
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            return model
            
        except ImportError:
            return None
    
    def fit(
        self,
        data: TimeSeriesData,
        sequence_length: int = 10,
        epochs: int = 50,
        hidden_units: Optional[List[int]] = None,
        **kwargs
    ) -> None:
        """
        Fit neural network model on training data.
        
        Args:
            data: Training time series data
            sequence_length: Number of timesteps for input sequence
            epochs: Number of training epochs
            hidden_units: List of units in hidden layers
        """
        if not self.validate_data(data):
            raise ValueError("Invalid training data")
        
        self.training_data = data
        self.sequence_length = min(sequence_length, data.length // 2)
        self.epochs = epochs
        self.hidden_units = hidden_units or [50, 25]
        
        # Normalize data
        values = np.array(data.values)
        self.min_val = float(values.min())
        self.max_val = float(values.max())
        
        if self.max_val - self.min_val > 0:
            normalized = (values - self.min_val) / (self.max_val - self.min_val)
        else:
            normalized = np.zeros_like(values)
        
        # Create sequences
        X, y = DataUtils.create_sequences(normalized.tolist(), self.sequence_length)
        
        if len(X) < 2:
            raise ValueError("Not enough data for sequence creation")
        
        # Create and train model
        self.model = self._create_model((self.sequence_length,), self.hidden_units)
        
        if self.model is None:
            raise ImportError("TensorFlow/Keras not available")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y, epochs=self.epochs, batch_size=16, verbose=0)
    
    def predict(self, steps: int, **kwargs) -> ForecastResult:
        """
        Generate neural network predictions.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            ForecastResult with predictions
        """
        start_time = time.time()
        
        if self.model is None or self.training_data is None:
            return ForecastResult.failed(
                model_name="NeuralNetwork",
                message="Model not fitted. Call fit() first."
            )
        
        try:
            # Get last sequence from training data
            values = np.array(self.training_data.values)
            
            if self.max_val - self.min_val > 0:
                normalized = (values - self.min_val) / (self.max_val - self.min_val)
            else:
                normalized = np.zeros_like(values)
            
            current_sequence = normalized[-self.sequence_length:].tolist()
            
            # Generate predictions iteratively
            predictions_normalized = []
            
            for _ in range(steps):
                # Prepare input
                X_input = np.array(current_sequence[-self.sequence_length:]).reshape(1, -1)
                
                # Predict next value
                next_pred = self.model.predict(X_input, verbose=0)[0, 0]
                predictions_normalized.append(float(next_pred))
                
                # Update sequence
                current_sequence.append(float(next_pred))
            
            # Denormalize predictions
            predictions = [
                p * (self.max_val - self.min_val) + self.min_val
                for p in predictions_normalized
            ]
            
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
                model_name="NeuralNetwork",
                message=f"Generated {steps} step forecast",
                predictions=predictions,
                prediction_timestamps=future_timestamps,
                model_params=self.get_model_params(),
                processing_time=processing_time
            )
            
        except Exception as e:
            return ForecastResult.failed(
                model_name="NeuralNetwork",
                message=f"Forecasting failed: {str(e)}"
            )
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get neural network parameters."""
        return {
            "sequence_length": self.sequence_length,
            "epochs": self.epochs,
            "hidden_units": self.hidden_units,
            "normalization": {
                "min": self.min_val,
                "max": self.max_val
            }
        }
