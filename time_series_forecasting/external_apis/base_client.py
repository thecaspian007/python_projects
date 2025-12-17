"""
Abstract base class for external API clients.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import requests


class BaseAPIClient(ABC):
    """Abstract base class for external API clients."""
    
    def __init__(self, base_url: str):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self._setup_headers()
    
    def _setup_headers(self):
        """Setup default headers for requests."""
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "TimeSeriesForecasting/1.0"
        })
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> requests.Response:
        """
        Make HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Query parameters
            timeout: Request timeout in seconds
            
        Returns:
            Response object
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}" if endpoint else self.base_url
        
        response = self.session.request(
            method=method,
            url=url,
            params=params,
            timeout=timeout
        )
        
        return response
    
    def get(self, endpoint: str = "", params: Optional[Dict[str, Any]] = None, **kwargs) -> requests.Response:
        """Make GET request."""
        return self._make_request("GET", endpoint, params=params, **kwargs)
