"""
Utility functions for image I/O operations.
"""
import os
import cv2
import numpy as np
import base64
import uuid
from typing import Tuple, Optional
from io import BytesIO
from PIL import Image
import requests

from config.settings import settings


class ImageUtils:
    """Utility class for image operations."""
    
    @staticmethod
    def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        """Read image from bytes."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    
    @staticmethod
    def read_image_from_file(filepath: str) -> np.ndarray:
        """Read image from file path."""
        return cv2.imread(filepath)
    
    @staticmethod
    def save_image(image: np.ndarray, filename: Optional[str] = None) -> str:
        """Save image to output directory and return the path."""
        if filename is None:
            filename = f"{uuid.uuid4().hex}.png"
        
        filepath = os.path.join(settings.OUTPUT_DIR, filename)
        cv2.imwrite(filepath, image)
        return filepath
    
    @staticmethod
    def image_to_bytes(image: np.ndarray, format: str = ".png") -> bytes:
        """Convert numpy image to bytes."""
        _, buffer = cv2.imencode(format, image)
        return buffer.tobytes()
    
    @staticmethod
    def image_to_base64(image: np.ndarray, format: str = ".png") -> str:
        """Convert numpy image to base64 string."""
        img_bytes = ImageUtils.image_to_bytes(image, format)
        return base64.b64encode(img_bytes).decode("utf-8")
    
    @staticmethod
    def base64_to_image(base64_str: str) -> np.ndarray:
        """Convert base64 string to numpy image."""
        img_bytes = base64.b64decode(base64_str)
        return ImageUtils.read_image_from_bytes(img_bytes)
    
    @staticmethod
    def download_image_from_url(url: str) -> np.ndarray:
        """Download image from URL and return as numpy array."""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return ImageUtils.read_image_from_bytes(response.content)
    
    @staticmethod
    def save_image_from_url(url: str, filename: Optional[str] = None) -> str:
        """Download image from URL and save to output directory."""
        image = ImageUtils.download_image_from_url(url)
        return ImageUtils.save_image(image, filename)
    
    @staticmethod
    def is_valid_extension(filename: str) -> bool:
        """Check if file has a valid image extension."""
        if not filename:
            return False
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return ext in settings.ALLOWED_EXTENSIONS
    
    @staticmethod
    def get_image_dimensions(image: np.ndarray) -> Tuple[int, int]:
        """Get image dimensions (height, width)."""
        return image.shape[:2]
    
    @staticmethod
    def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize image to specified dimensions."""
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format."""
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL format."""
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
