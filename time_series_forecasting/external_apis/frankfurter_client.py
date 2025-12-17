"""
Frankfurter API client for currency exchange rate data.

Frankfurter is a free, open-source API for currency exchange rates.
It tracks rates published by the European Central Bank.
No authentication required.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from .base_client import BaseAPIClient
from config.settings import settings
from models.time_series_data import TimeSeriesData


class FrankfurterClient(BaseAPIClient):
    """
    Client for Frankfurter API.
    
    Frankfurter provides:
    - Current and historical exchange rates
    - Currency conversion
    - Rates from European Central Bank
    - No authentication required
    
    API Documentation: https://www.frankfurter.app/docs/
    """
    
    def __init__(self):
        """Initialize Frankfurter client."""
        super().__init__(settings.FRANKFURTER_BASE_URL)
    
    def get_latest_rates(
        self,
        base: str = "EUR",
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get latest exchange rates.
        
        Args:
            base: Base currency code
            symbols: List of target currency codes
            
        Returns:
            API response with rates
        """
        params = {"from": base}
        if symbols:
            params["to"] = ",".join(symbols)
        
        response = self.get("latest", params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_historical_rates(
        self,
        start_date: str,
        end_date: str,
        base: str = "EUR",
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get historical exchange rates for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            base: Base currency code
            symbols: List of target currency codes
            
        Returns:
            API response with historical rates
        """
        params = {"from": base}
        if symbols:
            params["to"] = ",".join(symbols)
        
        endpoint = f"{start_date}..{end_date}"
        
        response = self.get(endpoint, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_rate_on_date(
        self,
        date: str,
        base: str = "EUR",
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get exchange rate for a specific date.
        
        Args:
            date: Date (YYYY-MM-DD)
            base: Base currency code
            symbols: List of target currency codes
            
        Returns:
            API response with rates
        """
        params = {"from": base}
        if symbols:
            params["to"] = ",".join(symbols)
        
        response = self.get(date, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_exchange_rate_series(
        self,
        base: str = "EUR",
        target: str = "USD",
        days: int = 30
    ) -> TimeSeriesData:
        """
        Get exchange rate time series.
        
        Args:
            base: Base currency code
            target: Target currency code
            days: Number of days of historical data
            
        Returns:
            TimeSeriesData with exchange rates
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        response = self.get_historical_rates(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            base=base,
            symbols=[target]
        )
        
        rates_data = response.get("rates", {})
        
        # Sort dates and extract values
        sorted_dates = sorted(rates_data.keys())
        timestamps = sorted_dates
        values = [rates_data[date].get(target, 0) for date in sorted_dates]
        
        return TimeSeriesData(
            name=f"{base}_{target}_rate",
            values=values,
            timestamps=timestamps,
            source="Frankfurter",
            metadata={
                "base_currency": base,
                "target_currency": target,
                "data_source": "European Central Bank"
            }
        )
    
    def get_available_currencies(self) -> Dict[str, str]:
        """
        Get list of available currencies.
        
        Returns:
            Dictionary of currency codes and names
        """
        response = self.get("currencies")
        response.raise_for_status()
        return response.json()
    
    def convert_currency(
        self,
        amount: float,
        from_currency: str,
        to_currency: str,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert amount between currencies.
        
        Args:
            amount: Amount to convert
            from_currency: Source currency code
            to_currency: Target currency code
            date: Optional date for historical conversion
            
        Returns:
            Conversion result
        """
        endpoint = date if date else "latest"
        params = {
            "amount": amount,
            "from": from_currency,
            "to": to_currency
        }
        
        response = self.get(endpoint, params=params)
        response.raise_for_status()
        
        return response.json()
