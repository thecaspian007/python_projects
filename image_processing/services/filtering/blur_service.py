"""
Blur filtering service for image processing.
"""
import cv2
import numpy as np
from typing import Dict, Any
import time

from services.base_processor import BaseProcessor
from models.image_result import ImageResult
from utils.image_utils import ImageUtils


class BlurService(BaseProcessor):
    """
    Service for applying various blur filters to images.
    
    Supported blur methods:
    - Gaussian Blur: Smooths images using Gaussian kernel
    - Median Blur: Good for salt-and-pepper noise
    - Bilateral Filter: Edge-preserving smoothing
    - Box Blur: Simple averaging filter
    - Motion Blur: Simulates motion blur effect
    """
    
    BLUR_METHODS = {
        "gaussian": "Gaussian blur using Gaussian kernel",
        "median": "Median filter for salt-and-pepper noise",
        "bilateral": "Edge-preserving bilateral filter",
        "box": "Simple box/averaging filter",
        "motion": "Simulated motion blur effect"
    }
    
    def __init__(self):
        super().__init__("BlurService")
    
    def get_supported_options(self) -> Dict[str, Any]:
        """Get supported blur options."""
        return {
            "method": {
                "description": "Blur method to use",
                "options": list(self.BLUR_METHODS.keys()),
                "default": "gaussian"
            },
            "kernel_size": {
                "description": "Size of the blur kernel (must be odd)",
                "default": 5,
                "min": 3,
                "max": 51
            },
            "sigma": {
                "description": "Sigma value for Gaussian blur",
                "default": 0
            }
        }
    
    def process(
        self,
        image: np.ndarray,
        method: str = "gaussian",
        kernel_size: int = 5,
        sigma: float = 0,
        **kwargs
    ) -> ImageResult:
        """
        Apply blur filter to image.
        
        Args:
            image: Input image as numpy array
            method: Blur method (gaussian, median, bilateral, box, motion)
            kernel_size: Size of blur kernel (must be odd)
            sigma: Sigma for Gaussian blur
            
        Returns:
            ImageResult with processed image path
        """
        start_time = time.time()
        
        if not self.validate_image(image):
            return ImageResult.failed(
                operation="blur",
                message="Invalid input image"
            )
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        try:
            if method == "gaussian":
                blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            
            elif method == "median":
                blurred = cv2.medianBlur(image, kernel_size)
            
            elif method == "bilateral":
                d = kernel_size
                sigma_color = kwargs.get("sigma_color", 75)
                sigma_space = kwargs.get("sigma_space", 75)
                blurred = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            
            elif method == "box":
                blurred = cv2.blur(image, (kernel_size, kernel_size))
            
            elif method == "motion":
                # Create motion blur kernel
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
                kernel = kernel / kernel_size
                blurred = cv2.filter2D(image, -1, kernel)
            
            else:
                return ImageResult.failed(
                    operation="blur",
                    message=f"Unknown blur method: {method}. Supported: {list(self.BLUR_METHODS.keys())}"
                )
            
            # Save the processed image
            output_path = ImageUtils.save_image(blurred)
            processing_time = time.time() - start_time
            
            return ImageResult.success(
                operation="blur",
                message=f"Successfully applied {method} blur",
                output_path=output_path,
                processing_time=processing_time,
                metadata={
                    "method": method,
                    "kernel_size": kernel_size,
                    "sigma": sigma
                }
            )
            
        except Exception as e:
            return ImageResult.failed(
                operation="blur",
                message=f"Blur processing failed: {str(e)}"
            )
    
    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> ImageResult:
        """Apply Gaussian blur."""
        return self.process(image, method="gaussian", kernel_size=kernel_size, sigma=sigma)
    
    def median_blur(self, image: np.ndarray, kernel_size: int = 5) -> ImageResult:
        """Apply median blur."""
        return self.process(image, method="median", kernel_size=kernel_size)
    
    def bilateral_filter(
        self,
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75
    ) -> ImageResult:
        """Apply bilateral filter."""
        return self.process(
            image,
            method="bilateral",
            kernel_size=d,
            sigma_color=sigma_color,
            sigma_space=sigma_space
        )
