"""
Open-Meteo API client for historical weather data.

Open-Meteo is a free weather API that requires no authentication.
It provides historical weather data for any location worldwide.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd

from .base_client import BaseAPIClient
from config.settings import settings
from models.time_series_data import TimeSeriesData


class OpenMeteoClient(BaseAPIClient):
    """
    Client for Open-Meteo API.
    
    Open-Meteo provides:
    - Historical weather data (temperature, precipitation, wind, etc.)
    - Global coverage
    - No authentication required
    - Free access
    
    API Documentation: https://open-meteo.com/en/docs/historical-weather-api
    """
    
    WEATHER_VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_10m",
        "wind_direction_10m",
        "surface_pressure"
    ]
    
    def __init__(self):
        """Initialize Open-Meteo client."""
        super().__init__(settings.OPEN_METEO_BASE_URL)
    
    def get_historical_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None,
        timezone: str = "auto"
    ) -> Dict[str, Any]:
        """
        Get historical weather data for a location.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: List of weather variables to fetch
            timezone: Timezone (default: auto)
            
        Returns:
            API response with weather data
        """
        if variables is None:
            variables = ["temperature_2m"]
        
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(variables) if "temperature_2m" not in variables else None,
            "hourly": ",".join(variables),
            "timezone": timezone
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        response = self.get("archive", params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_temperature_series(
        self,
        location: str = "new_york",
        days: int = 30,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None
    ) -> TimeSeriesData:
        """
        Get temperature time series for a location.
        
        Args:
            location: Location name from predefined list
            days: Number of days of historical data
            latitude: Custom latitude (overrides location)
            longitude: Custom longitude (overrides location)
            
        Returns:
            TimeSeriesData with temperature values
        """
        # Get coordinates
        if latitude is None or longitude is None:
            if location in settings.DEFAULT_LOCATIONS:
                coords = settings.DEFAULT_LOCATIONS[location]
                latitude = coords["latitude"]
                longitude = coords["longitude"]
            else:
                raise ValueError(f"Unknown location: {location}. Available: {list(settings.DEFAULT_LOCATIONS.keys())}")
        
        # Calculate date range
        end_date = datetime.now() - timedelta(days=5)  # API has 5-day delay
        start_date = end_date - timedelta(days=days)
        
        # Fetch data
        response = self.get_historical_weather(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            variables=["temperature_2m"]
        )
        
        # Extract hourly data
        hourly_data = response.get("hourly", {})
        timestamps = hourly_data.get("time", [])
        temperatures = hourly_data.get("temperature_2m", [])
        
        # Handle None values
        temperatures = [t if t is not None else 0 for t in temperatures]
        
        return TimeSeriesData(
            name=f"temperature_{location}",
            values=temperatures,
            timestamps=timestamps,
            source="Open-Meteo",
            metadata={
                "location": location,
                "latitude": latitude,
                "longitude": longitude,
                "unit": "°C",
                "variable": "temperature_2m"
            }
        )
    
    def get_precipitation_series(
        self,
        location: str = "new_york",
        days: int = 30
    ) -> TimeSeriesData:
        """Get precipitation time series for a location."""
        if location not in settings.DEFAULT_LOCATIONS:
            raise ValueError(f"Unknown location: {location}")
        
        coords = settings.DEFAULT_LOCATIONS[location]
        
        end_date = datetime.now() - timedelta(days=5)
        start_date = end_date - timedelta(days=days)
        
        response = self.get_historical_weather(
            latitude=coords["latitude"],
            longitude=coords["longitude"],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            variables=["precipitation"]
        )
        
        hourly_data = response.get("hourly", {})
        timestamps = hourly_data.get("time", [])
        precipitation = hourly_data.get("precipitation", [])
        
        precipitation = [p if p is not None else 0 for p in precipitation]
        
        return TimeSeriesData(
            name=f"precipitation_{location}",
            values=precipitation,
            timestamps=timestamps,
            source="Open-Meteo",
            metadata={
                "location": location,
                "unit": "mm",
                "variable": "precipitation"
            }
        )
    
    def get_available_locations(self) -> Dict[str, Dict[str, float]]:
        """Get list of predefined locations."""
        return settings.DEFAULT_LOCATIONS
    
    def get_available_variables(self) -> List[str]:
        """Get list of available weather variables."""
        return self.WEATHER_VARIABLES
