"""
Utility functions for data preprocessing and manipulation.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime, timedelta


class DataUtils:
    """Utility class for data preprocessing."""
    
    @staticmethod
    def normalize_data(data: List[float]) -> Tuple[List[float], float, float]:
        """
        Normalize data using min-max scaling.
        
        Args:
            data: List of values to normalize
            
        Returns:
            Tuple of (normalized_data, min_val, max_val)
        """
        arr = np.array(data)
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        
        if max_val - min_val == 0:
            normalized = np.zeros_like(arr)
        else:
            normalized = (arr - min_val) / (max_val - min_val)
        
        return normalized.tolist(), min_val, max_val
    
    @staticmethod
    def denormalize_data(
        normalized_data: List[float],
        min_val: float,
        max_val: float
    ) -> List[float]:
        """
        Denormalize data back to original scale.
        
        Args:
            normalized_data: Normalized values
            min_val: Original minimum value
            max_val: Original maximum value
            
        Returns:
            Denormalized values
        """
        arr = np.array(normalized_data)
        denormalized = arr * (max_val - min_val) + min_val
        return denormalized.tolist()
    
    @staticmethod
    def standardize_data(data: List[float]) -> Tuple[List[float], float, float]:
        """
        Standardize data using z-score normalization.
        
        Args:
            data: List of values to standardize
            
        Returns:
            Tuple of (standardized_data, mean, std)
        """
        arr = np.array(data)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        
        if std == 0:
            standardized = np.zeros_like(arr)
        else:
            standardized = (arr - mean) / std
        
        return standardized.tolist(), mean, std
    
    @staticmethod
    def destandardize_data(
        standardized_data: List[float],
        mean: float,
        std: float
    ) -> List[float]:
        """Destandardize data back to original scale."""
        arr = np.array(standardized_data)
        destandardized = arr * std + mean
        return destandardized.tolist()
    
    @staticmethod
    def create_sequences(
        data: List[float],
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series forecasting.
        
        Args:
            data: List of values
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) where X is input sequences and y is targets
        """
        arr = np.array(data)
        X, y = [], []
        
        for i in range(len(arr) - sequence_length):
            X.append(arr[i:i + sequence_length])
            y.append(arr[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def fill_missing_values(
        data: List[Optional[float]],
        method: str = "linear"
    ) -> List[float]:
        """
        Fill missing values in time series.
        
        Args:
            data: List with potential None values
            method: Interpolation method (linear, forward, backward, mean)
            
        Returns:
            List with filled values
        """
        series = pd.Series(data)
        
        if method == "linear":
            filled = series.interpolate(method="linear")
        elif method == "forward":
            filled = series.fillna(method="ffill")
        elif method == "backward":
            filled = series.fillna(method="bfill")
        elif method == "mean":
            filled = series.fillna(series.mean())
        else:
            filled = series.interpolate(method="linear")
        
        # Fill any remaining NaN at edges
        filled = filled.fillna(method="ffill").fillna(method="bfill")
        
        return filled.tolist()
    
    @staticmethod
    def detect_outliers(
        data: List[float],
        method: str = "iqr",
        threshold: float = 1.5
    ) -> List[int]:
        """
        Detect outliers in time series.
        
        Args:
            data: List of values
            method: Detection method (iqr, zscore)
            threshold: Threshold for outlier detection
            
        Returns:
            List of indices of outliers
        """
        arr = np.array(data)
        outlier_indices = []
        
        if method == "iqr":
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            for i, val in enumerate(arr):
                if val < lower_bound or val > upper_bound:
                    outlier_indices.append(i)
        
        elif method == "zscore":
            mean = np.mean(arr)
            std = np.std(arr)
            
            for i, val in enumerate(arr):
                z_score = abs((val - mean) / std) if std > 0 else 0
                if z_score > threshold:
                    outlier_indices.append(i)
        
        return outlier_indices
    
    @staticmethod
    def calculate_differencing(data: List[float], periods: int = 1) -> List[float]:
        """
        Calculate differenced series for stationarity.
        
        Args:
            data: List of values
            periods: Number of periods to difference
            
        Returns:
            Differenced series
        """
        arr = np.array(data)
        differenced = np.diff(arr, n=periods)
        return differenced.tolist()
    
    @staticmethod
    def generate_future_timestamps(
        last_timestamp: str,
        periods: int,
        frequency: str = "D"
    ) -> List[str]:
        """
        Generate future timestamps for predictions.
        
        Args:
            last_timestamp: Last timestamp in the data
            periods: Number of periods to generate
            frequency: Frequency (D=daily, H=hourly, W=weekly)
            
        Returns:
            List of future timestamp strings
        """
        last_dt = pd.to_datetime(last_timestamp)
        
        freq_map = {
            "D": timedelta(days=1),
            "H": timedelta(hours=1),
            "W": timedelta(weeks=1),
            "M": timedelta(days=30)
        }
        
        delta = freq_map.get(frequency, timedelta(days=1))
        
        future_timestamps = []
        for i in range(1, periods + 1):
            future_dt = last_dt + (delta * i)
            future_timestamps.append(future_dt.isoformat())
        
        return future_timestamps
