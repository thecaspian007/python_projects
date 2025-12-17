"""
Abstract base class for all image processors.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

from models.image_result import ImageResult


class BaseProcessor(ABC):
    """Abstract base class for image processing services."""
    
    def __init__(self, name: str):
        """Initialize processor with a name."""
        self.name = name
    
    @abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> ImageResult:
        """
        Process an image.
        
        Args:
            image: Input image as numpy array
            **kwargs: Additional processing parameters
            
        Returns:
            ImageResult with processing outcome
        """
        pass
    
    @abstractmethod
    def get_supported_options(self) -> Dict[str, Any]:
        """
        Get supported processing options.
        
        Returns:
            Dictionary of option names and their descriptions/defaults
        """
        pass
    
    def validate_image(self, image: np.ndarray) -> bool:
        """Validate that the image is valid for processing."""
        if image is None:
            return False
        if not isinstance(image, np.ndarray):
            return False
        if len(image.shape) < 2:
            return False
        return True
