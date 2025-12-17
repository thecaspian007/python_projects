"""
Configuration settings for the Image Processing API.
"""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings."""
    
    # Flask settings
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    
    # DeepAI API settings
    DEEPAI_API_KEY: str = os.getenv("DEEPAI_API_KEY", "")
    DEEPAI_BASE_URL: str = "https://api.deepai.org/api"
    
    # File upload settings
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS: set = None
    
    # Output settings
    OUTPUT_DIR: str = "output"
    
    def __post_init__(self):
        if self.ALLOWED_EXTENSIONS is None:
            self.ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


# Global settings instance
settings = Settings()
