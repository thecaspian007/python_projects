"""
Abstract base class for external API clients.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import requests


class BaseAPIClient(ABC):
    """Abstract base class for external API clients."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self._setup_headers()
    
    def _setup_headers(self):
        """Setup default headers for requests."""
        self.session.headers.update({
            "Accept": "application/json"
        })
    
    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for the API."""
        pass
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 60
    ) -> requests.Response:
        """
        Make HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data
            files: Files to upload
            params: Query parameters
            timeout: Request timeout in seconds
            
        Returns:
            Response object
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_auth_headers()
        
        response = self.session.request(
            method=method,
            url=url,
            data=data,
            files=files,
            params=params,
            headers=headers,
            timeout=timeout
        )
        
        return response
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> requests.Response:
        """Make GET request."""
        return self._make_request("GET", endpoint, params=params, **kwargs)
    
    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> requests.Response:
        """Make POST request."""
        return self._make_request("POST", endpoint, data=data, files=files, **kwargs)
