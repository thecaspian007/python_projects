"""
DeepAI API client for image processing operations.
"""
from typing import Dict, Optional, Any, Union
from io import BytesIO
import requests

from .base_client import BaseAPIClient
from config.settings import settings


class DeepAIClient(BaseAPIClient):
    """
    Client for DeepAI API.
    
    DeepAI provides various AI-powered image processing endpoints including:
    - Image colorization
    - Super resolution (image upscaling)
    - Background removal
    - Neural style transfer
    """
    
    # API endpoints
    ENDPOINTS = {
        "colorizer": "colorizer",
        "super_resolution": "torch-srgan",
        "background_remover": "background-remover",
        "neural_style": "neural-style",
        "deepdream": "deepdream",
        "image_similarity": "image-similarity"
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize DeepAI client.
        
        Args:
            api_key: DeepAI API key. If not provided, uses DEEPAI_API_KEY from settings.
        """
        api_key = api_key or settings.DEEPAI_API_KEY
        super().__init__(settings.DEEPAI_BASE_URL, api_key)
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get DeepAI authentication headers."""
        if self.api_key:
            return {"api-key": self.api_key}
        return {}
    
    def _process_image(
        self,
        endpoint: str,
        image: Union[bytes, str],
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process image using specified endpoint.
        
        Args:
            endpoint: API endpoint name
            image: Image bytes or URL
            additional_params: Additional parameters for the API
            
        Returns:
            API response as dictionary
        """
        files = {}
        data = additional_params or {}
        
        if isinstance(image, bytes):
            files["image"] = ("image.jpg", BytesIO(image), "image/jpeg")
        else:
            # Assume it's a URL
            data["image"] = image
        
        response = self.post(endpoint, data=data, files=files if files else None)
        response.raise_for_status()
        
        return response.json()
    
    def colorize(self, image: Union[bytes, str]) -> Dict[str, Any]:
        """
        Colorize a black and white image.
        
        Args:
            image: Image bytes or URL
            
        Returns:
            API response with output_url
        """
        return self._process_image(self.ENDPOINTS["colorizer"], image)
    
    def super_resolution(self, image: Union[bytes, str]) -> Dict[str, Any]:
        """
        Upscale an image using AI super resolution.
        
        Args:
            image: Image bytes or URL
            
        Returns:
            API response with output_url
        """
        return self._process_image(self.ENDPOINTS["super_resolution"], image)
    
    def remove_background(self, image: Union[bytes, str]) -> Dict[str, Any]:
        """
        Remove background from an image.
        
        Args:
            image: Image bytes or URL
            
        Returns:
            API response with output_url
        """
        return self._process_image(self.ENDPOINTS["background_remover"], image)
    
    def neural_style_transfer(
        self,
        content_image: Union[bytes, str],
        style_image: Union[bytes, str]
    ) -> Dict[str, Any]:
        """
        Apply neural style transfer to an image.
        
        Args:
            content_image: Content image bytes or URL
            style_image: Style image bytes or URL
            
        Returns:
            API response with output_url
        """
        files = {}
        data = {}
        
        if isinstance(content_image, bytes):
            files["content"] = ("content.jpg", BytesIO(content_image), "image/jpeg")
        else:
            data["content"] = content_image
        
        if isinstance(style_image, bytes):
            files["style"] = ("style.jpg", BytesIO(style_image), "image/jpeg")
        else:
            data["style"] = style_image
        
        response = self.post(
            self.ENDPOINTS["neural_style"],
            data=data,
            files=files if files else None
        )
        response.raise_for_status()
        
        return response.json()
    
    def deep_dream(self, image: Union[bytes, str]) -> Dict[str, Any]:
        """
        Apply DeepDream effect to an image.
        
        Args:
            image: Image bytes or URL
            
        Returns:
            API response with output_url
        """
        return self._process_image(self.ENDPOINTS["deepdream"], image)
    
    def image_similarity(
        self,
        image1: Union[bytes, str],
        image2: Union[bytes, str]
    ) -> Dict[str, Any]:
        """
        Compare two images for similarity.
        
        Args:
            image1: First image bytes or URL
            image2: Second image bytes or URL
            
        Returns:
            API response with similarity score
        """
        files = {}
        data = {}
        
        if isinstance(image1, bytes):
            files["image1"] = ("image1.jpg", BytesIO(image1), "image/jpeg")
        else:
            data["image1"] = image1
        
        if isinstance(image2, bytes):
            files["image2"] = ("image2.jpg", BytesIO(image2), "image/jpeg")
        else:
            data["image2"] = image2
        
        response = self.post(
            self.ENDPOINTS["image_similarity"],
            data=data,
            files=files if files else None
        )
        response.raise_for_status()
        
        return response.json()
