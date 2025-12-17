"""
Configuration settings for the Time Series Forecasting API.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Settings:
    """Application settings."""
    
    # Flask settings
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 5001  # Different port from image processing
    
    # External API settings (no auth required)
    OPEN_METEO_BASE_URL: str = "https://archive-api.open-meteo.com/v1"
    FRANKFURTER_BASE_URL: str = "https://api.frankfurter.app"
    
    # Default locations for weather data
    DEFAULT_LOCATIONS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "new_york": {"latitude": 40.7128, "longitude": -74.0060},
        "london": {"latitude": 51.5074, "longitude": -0.1278},
        "tokyo": {"latitude": 35.6762, "longitude": 139.6503},
        "paris": {"latitude": 48.8566, "longitude": 2.3522},
        "sydney": {"latitude": -33.8688, "longitude": 151.2093},
        "mumbai": {"latitude": 19.0760, "longitude": 72.8777}
    })
    
    # Default currencies
    DEFAULT_CURRENCIES: list = field(default_factory=lambda: [
        "USD", "EUR", "GBP", "JPY", "INR", "AUD", "CAD", "CHF"
    ])
    
    # Model settings
    ARIMA_DEFAULT_ORDER: tuple = (5, 1, 0)
    ETS_DEFAULT_TREND: str = "add"
    ETS_DEFAULT_SEASONAL: str = None
    
    # Output settings
    OUTPUT_DIR: str = "output"
    
    def __post_init__(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


# Global settings instance
settings = Settings()
