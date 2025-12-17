"""
Data models for image processing results.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime


class ProcessingStatus(Enum):
    """Status of image processing operation."""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


@dataclass
class ImageResult:
    """Result of an image processing operation."""
    
    status: ProcessingStatus
    operation: str
    message: str
    output_url: Optional[str] = None
    output_path: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "operation": self.operation,
            "message": self.message,
            "output_url": self.output_url,
            "output_path": self.output_path,
            "processing_time": self.processing_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def success(cls, operation: str, message: str, **kwargs) -> "ImageResult":
        """Create a success result."""
        return cls(
            status=ProcessingStatus.SUCCESS,
            operation=operation,
            message=message,
            **kwargs
        )
    
    @classmethod
    def failed(cls, operation: str, message: str, **kwargs) -> "ImageResult":
        """Create a failed result."""
        return cls(
            status=ProcessingStatus.FAILED,
            operation=operation,
            message=message,
            **kwargs
        )
