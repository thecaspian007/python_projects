"""
Super Resolution service using DeepAI API.
"""
import time
from typing import Dict, Any
import numpy as np

from services.base_processor import BaseProcessor
from models.image_result import ImageResult
from utils.image_utils import ImageUtils
from external_apis.deepai_client import DeepAIClient


class SuperResolutionService(BaseProcessor):
    """
    Service for upscaling images using AI super resolution via DeepAI API.
    
    This service uses DeepAI's torch-srgan endpoint to upscale images
    while maintaining quality using deep learning models.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize super resolution service.
        
        Args:
            api_key: DeepAI API key. If not provided, uses environment variable.
        """
        super().__init__("SuperResolutionService")
        self.client = DeepAIClient(api_key)
    
    def get_supported_options(self) -> Dict[str, Any]:
        """Get supported super resolution options."""
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
        Upscale an image using AI super resolution.
        
        Args:
            image: Input image as numpy array
            save_local: Whether to save the result locally
            
        Returns:
            ImageResult with upscaled image URL/path
        """
        start_time = time.time()
        
        if not self.validate_image(image):
            return ImageResult.failed(
                operation="super_resolution",
                message="Invalid input image"
            )
        
        try:
            # Get original dimensions
            original_height, original_width = ImageUtils.get_image_dimensions(image)
            
            # Convert image to bytes
            image_bytes = ImageUtils.image_to_bytes(image, ".jpg")
            
            # Call DeepAI API
            response = self.client.super_resolution(image_bytes)
            
            if "output_url" not in response:
                error_msg = response.get("err", "Unknown error from DeepAI API")
                return ImageResult.failed(
                    operation="super_resolution",
                    message=f"API error: {error_msg}"
                )
            
            output_url = response["output_url"]
            output_path = None
            new_dimensions = None
            
            # Download and save locally if requested
            if save_local:
                output_path = ImageUtils.save_image_from_url(output_url)
                # Get new dimensions
                upscaled_image = ImageUtils.read_image_from_file(output_path)
                if upscaled_image is not None:
                    new_height, new_width = ImageUtils.get_image_dimensions(upscaled_image)
                    new_dimensions = {"width": new_width, "height": new_height}
            
            processing_time = time.time() - start_time
            
            return ImageResult.success(
                operation="super_resolution",
                message="Successfully upscaled image",
                output_url=output_url,
                output_path=output_path,
                processing_time=processing_time,
                metadata={
                    "api": "DeepAI",
                    "endpoint": "torch-srgan",
                    "original_size": {"width": original_width, "height": original_height},
                    "new_size": new_dimensions
                }
            )
            
        except Exception as e:
            return ImageResult.failed(
                operation="super_resolution",
                message=f"Super resolution failed: {str(e)}"
            )
    
    def upscale_from_url(self, image_url: str, save_local: bool = True) -> ImageResult:
        """
        Upscale an image from URL.
        
        Args:
            image_url: URL of the image to upscale
            save_local: Whether to save the result locally
            
        Returns:
            ImageResult with upscaled image URL/path
        """
        start_time = time.time()
        
        try:
            # Call DeepAI API with URL
            response = self.client.super_resolution(image_url)
            
            if "output_url" not in response:
                error_msg = response.get("err", "Unknown error from DeepAI API")
                return ImageResult.failed(
                    operation="super_resolution",
                    message=f"API error: {error_msg}"
                )
            
            output_url = response["output_url"]
            output_path = None
            
            if save_local:
                output_path = ImageUtils.save_image_from_url(output_url)
            
            processing_time = time.time() - start_time
            
            return ImageResult.success(
                operation="super_resolution",
                message="Successfully upscaled image",
                output_url=output_url,
                output_path=output_path,
                processing_time=processing_time,
                metadata={
                    "api": "DeepAI",
                    "endpoint": "torch-srgan",
                    "source": "url"
                }
            )
            
        except Exception as e:
            return ImageResult.failed(
                operation="super_resolution",
                message=f"Super resolution failed: {str(e)}"
            )
