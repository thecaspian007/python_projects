"""
Colorization service using DeepAI API.
"""
import time
from typing import Dict, Any, Union
import numpy as np

from services.base_processor import BaseProcessor
from models.image_result import ImageResult
from utils.image_utils import ImageUtils
from external_apis.deepai_client import DeepAIClient


class ColorizationService(BaseProcessor):
    """
    Service for colorizing black and white images using DeepAI API.
    
    This service uses DeepAI's colorizer endpoint to add realistic colors
    to grayscale or black-and-white images using neural networks.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize colorization service.
        
        Args:
            api_key: DeepAI API key. If not provided, uses environment variable.
        """
        super().__init__("ColorizationService")
        self.client = DeepAIClient(api_key)
    
    def get_supported_options(self) -> Dict[str, Any]:
        """Get supported colorization options."""
        return {
            "save_local": {
                "description": "Whether to download and save the result locally",
                "default": True
            }
        }
    
    def process(
        self,
        image: np.ndarray,
        save_local: bool = True,
        **kwargs
    ) -> ImageResult:
        """
        Colorize a black and white image.
        
        Args:
            image: Input image as numpy array (grayscale or color)
            save_local: Whether to save the result locally
            
        Returns:
            ImageResult with colorized image URL/path
        """
        start_time = time.time()
        
        if not self.validate_image(image):
            return ImageResult.failed(
                operation="colorize",
                message="Invalid input image"
            )
        
        try:
            # Convert image to bytes
            image_bytes = ImageUtils.image_to_bytes(image, ".jpg")
            
            # Call DeepAI API
            response = self.client.colorize(image_bytes)
            
            if "output_url" not in response:
                error_msg = response.get("err", "Unknown error from DeepAI API")
                return ImageResult.failed(
                    operation="colorize",
                    message=f"API error: {error_msg}"
                )
            
            output_url = response["output_url"]
            output_path = None
            
            # Download and save locally if requested
            if save_local:
                output_path = ImageUtils.save_image_from_url(output_url)
            
            processing_time = time.time() - start_time
            
            return ImageResult.success(
                operation="colorize",
                message="Successfully colorized image",
                output_url=output_url,
                output_path=output_path,
                processing_time=processing_time,
                metadata={"api": "DeepAI", "endpoint": "colorizer"}
            )
            
        except Exception as e:
            return ImageResult.failed(
                operation="colorize",
                message=f"Colorization failed: {str(e)}"
            )
    
    def colorize_from_url(self, image_url: str, save_local: bool = True) -> ImageResult:
        """
        Colorize an image from URL.
        
        Args:
            image_url: URL of the image to colorize
            save_local: Whether to save the result locally
            
        Returns:
            ImageResult with colorized image URL/path
        """
        start_time = time.time()
        
        try:
            # Call DeepAI API with URL
            response = self.client.colorize(image_url)
            
            if "output_url" not in response:
                error_msg = response.get("err", "Unknown error from DeepAI API")
                return ImageResult.failed(
                    operation="colorize",
                    message=f"API error: {error_msg}"
                )
            
            output_url = response["output_url"]
            output_path = None
            
            if save_local:
                output_path = ImageUtils.save_image_from_url(output_url)
            
            processing_time = time.time() - start_time
            
            return ImageResult.success(
                operation="colorize",
                message="Successfully colorized image",
                output_url=output_url,
                output_path=output_path,
                processing_time=processing_time,
                metadata={
                    "api": "DeepAI",
                    "endpoint": "colorizer",
                    "source": "url"
                }
            )
            
        except Exception as e:
            return ImageResult.failed(
                operation="colorize",
                message=f"Colorization failed: {str(e)}"
            )
