"""
Data models for forecast results.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ForecastStatus(Enum):
    """Status of forecasting operation."""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


@dataclass
class ForecastResult:
    """Result of a forecasting operation."""
    
    status: ForecastStatus
    model_name: str
    message: str
    predictions: Optional[List[float]] = None
    prediction_timestamps: Optional[List[str]] = None
    actual_values: Optional[List[float]] = None
    metrics: Optional[Dict[str, float]] = None
    model_params: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "model_name": self.model_name,
            "message": self.message,
            "predictions": self.predictions,
            "prediction_timestamps": self.prediction_timestamps,
            "actual_values": self.actual_values,
            "metrics": self.metrics,
            "model_params": self.model_params,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def success(
        cls,
        model_name: str,
        message: str,
        predictions: List[float],
        **kwargs
    ) -> "ForecastResult":
        """Create a success result."""
        return cls(
            status=ForecastStatus.SUCCESS,
            model_name=model_name,
            message=message,
            predictions=predictions,
            **kwargs
        )
    
    @classmethod
    def failed(cls, model_name: str, message: str, **kwargs) -> "ForecastResult":
        """Create a failed result."""
        return cls(
            status=ForecastStatus.FAILED,
            model_name=model_name,
            message=message,
            **kwargs
        )


@dataclass
class EvaluationResult:
    """Result of model evaluation."""
    
    model_name: str
    metrics: Dict[str, float]
    predictions: List[float]
    actual_values: List[float]
    residuals: List[float] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def __post_init__(self):
        """Calculate residuals if not provided."""
        if not self.residuals and self.predictions and self.actual_values:
            self.residuals = [
                actual - pred 
                for actual, pred in zip(self.actual_values, self.predictions)
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "predictions": self.predictions,
            "actual_values": self.actual_values,
            "residuals": self.residuals,
            "timestamp": self.timestamp
        }
