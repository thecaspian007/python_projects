"""
Denoising service for image processing.
"""
import cv2
import numpy as np
from typing import Dict, Any
import time

from services.base_processor import BaseProcessor
from models.image_result import ImageResult
from utils.image_utils import ImageUtils


class DenoiseService(BaseProcessor):
    """
    Service for denoising images using various techniques.
    
    Supported denoising methods:
    - Non-Local Means: Best for general denoising
    - Bilateral Filter: Edge-preserving denoising
    - Morphological: Opens/closes for noise removal
    - Adaptive: Adaptive thresholding based
    """
    
    DENOISE_METHODS = {
        "nlm": "Non-Local Means denoising",
        "nlm_color": "Non-Local Means for color images",
        "bilateral": "Bilateral filter denoising",
        "morphological": "Morphological opening/closing",
        "adaptive": "Adaptive threshold denoising"
    }
    
    def __init__(self):
        super().__init__("DenoiseService")
    
    def get_supported_options(self) -> Dict[str, Any]:
        """Get supported denoising options."""
        return {
            "method": {
                "description": "Denoising method to use",
                "options": list(self.DENOISE_METHODS.keys()),
                "default": "nlm_color"
            },
            "h": {
                "description": "Filter strength (higher removes more noise but may lose detail)",
                "default": 10,
                "min": 1,
                "max": 30
            },
            "template_window_size": {
                "description": "Template window size for NLM (must be odd)",
                "default": 7
            },
            "search_window_size": {
                "description": "Search window size for NLM (must be odd)",
                "default": 21
            }
        }
    
    def process(
        self,
        image: np.ndarray,
        method: str = "nlm_color",
        h: float = 10,
        template_window_size: int = 7,
        search_window_size: int = 21,
        **kwargs
    ) -> ImageResult:
        """
        Apply denoising to image.
        
        Args:
            image: Input image as numpy array
            method: Denoising method
            h: Filter strength
            template_window_size: Template window size for NLM
            search_window_size: Search window size for NLM
            
        Returns:
            ImageResult with denoised image path
        """
        start_time = time.time()
        
        if not self.validate_image(image):
            return ImageResult.failed(
                operation="denoise",
                message="Invalid input image"
            )
        
        try:
            if method == "nlm":
                # For grayscale images
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                denoised = cv2.fastNlMeansDenoising(
                    gray, h=h,
                    templateWindowSize=template_window_size,
                    searchWindowSize=search_window_size
                )
            
            elif method == "nlm_color":
                # For color images
                if len(image.shape) == 2:
                    # Convert grayscale to color
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                denoised = cv2.fastNlMeansDenoisingColored(
                    image, h=h, hForColorComponents=h,
                    templateWindowSize=template_window_size,
                    searchWindowSize=search_window_size
                )
            
            elif method == "bilateral":
                d = kwargs.get("d", 9)
                sigma_color = kwargs.get("sigma_color", 75)
                sigma_space = kwargs.get("sigma_space", 75)
                denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            
            elif method == "morphological":
                kernel_size = kwargs.get("kernel_size", 5)
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                )
                # Opening followed by closing
                opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
                denoised = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            
            elif method == "adaptive":
                # Convert to grayscale if color
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                # Apply Gaussian blur first
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Apply adaptive threshold
                denoised = cv2.adaptiveThreshold(
                    blurred, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11, 2
                )
            
            else:
                return ImageResult.failed(
                    operation="denoise",
                    message=f"Unknown method: {method}. Supported: {list(self.DENOISE_METHODS.keys())}"
                )
            
            # Save the processed image
            output_path = ImageUtils.save_image(denoised)
            processing_time = time.time() - start_time
            
            return ImageResult.success(
                operation="denoise",
                message=f"Successfully applied {method} denoising",
                output_path=output_path,
                processing_time=processing_time,
                metadata={
                    "method": method,
                    "h": h,
                    "template_window_size": template_window_size,
                    "search_window_size": search_window_size
                }
            )
            
        except Exception as e:
            return ImageResult.failed(
                operation="denoise",
                message=f"Denoising failed: {str(e)}"
            )
    
    def nlm_denoise(self, image: np.ndarray, h: float = 10) -> ImageResult:
        """Apply Non-Local Means denoising."""
        return self.process(image, method="nlm_color", h=h)
    
    def bilateral_denoise(
        self,
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75
    ) -> ImageResult:
        """Apply bilateral filter denoising."""
        return self.process(
            image,
            method="bilateral",
            d=d,
            sigma_color=sigma_color,
            sigma_space=sigma_space
        )
