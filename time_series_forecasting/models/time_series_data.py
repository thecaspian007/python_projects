"""
Data models for time series data containers.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import numpy as np
import pandas as pd


@dataclass
class TimeSeriesData:
    """Container for time series data."""
    
    name: str
    values: List[float]
    timestamps: List[str]
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization."""
        if len(self.values) != len(self.timestamps):
            raise ValueError("Values and timestamps must have the same length")
    
    @property
    def length(self) -> int:
        """Get the number of data points."""
        return len(self.values)
    
    @property
    def values_array(self) -> np.ndarray:
        """Get values as numpy array."""
        return np.array(self.values)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            "timestamp": pd.to_datetime(self.timestamps),
            "value": self.values
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "values": self.values,
            "timestamps": self.timestamps,
            "source": self.source,
            "length": self.length,
            "metadata": self.metadata,
            "statistics": self.get_statistics()
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate basic statistics."""
        arr = self.values_array
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr))
        }
    
    def split_train_test(self, train_ratio: float = 0.8) -> tuple:
        """
        Split data into train and test sets.
        
        Args:
            train_ratio: Ratio of training data
            
        Returns:
            Tuple of (train_data, test_data) as TimeSeriesData objects
        """
        split_idx = int(len(self.values) * train_ratio)
        
        train_data = TimeSeriesData(
            name=f"{self.name}_train",
            values=self.values[:split_idx],
            timestamps=self.timestamps[:split_idx],
            source=self.source,
            metadata={**self.metadata, "split": "train"}
        )
        
        test_data = TimeSeriesData(
            name=f"{self.name}_test",
            values=self.values[split_idx:],
            timestamps=self.timestamps[split_idx:],
            source=self.source,
            metadata={**self.metadata, "split": "test"}
        )
        
        return train_data, test_data
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: str,
        source: str,
        value_column: str = "value",
        timestamp_column: str = "timestamp"
    ) -> "TimeSeriesData":
        """Create TimeSeriesData from pandas DataFrame."""
        return cls(
            name=name,
            values=df[value_column].tolist(),
            timestamps=df[timestamp_column].astype(str).tolist(),
            source=source
        )
