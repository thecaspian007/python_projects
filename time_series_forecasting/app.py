"""
Time Series Forecasting API Application.

A Flask-based REST API for time series data fetching, forecasting, and evaluation.

Features:
- Fetch live data from Open-Meteo (weather) and Frankfurter (currency) APIs
- Multiple forecasting models: ARIMA, ETS, Moving Average, Linear Regression, Neural Network
- Comprehensive evaluation metrics: RMSE, MAE, MAPE, R²

Usage:
    python app.py

The API will start on http://localhost:5001
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify
from flask_cors import CORS

from config.settings import settings
from controllers.forecast_controller import forecast_bp


def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__)
    
    # Configuration
    app.config["DEBUG"] = settings.DEBUG
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(forecast_bp)
    
    # Root endpoint
    @app.route("/")
    def index():
        return jsonify({
            "name": "Time Series Forecasting API",
            "version": "1.0.0",
            "description": "REST API for time series forecasting and analysis",
            "endpoints": {
                "health": "/api/v1/health",
                "datasets": {
                    "weather": "/api/v1/datasets/weather",
                    "currency": "/api/v1/datasets/currency",
                    "locations": "/api/v1/datasets/locations",
                    "currencies": "/api/v1/datasets/currencies"
                },
                "forecast": {
                    "arima": "/api/v1/forecast/arima",
                    "ets": "/api/v1/forecast/ets",
                    "moving_average": "/api/v1/forecast/moving-average",
                    "linear_regression": "/api/v1/forecast/linear-regression",
                    "neural_network": "/api/v1/forecast/neural-network"
                },
                "evaluate": {
                    "single": "/api/v1/evaluate",
                    "compare": "/api/v1/evaluate/compare",
                    "metrics_info": "/api/v1/metrics"
                },
                "pipelines": {
                    "weather_forecast": "/api/v1/pipeline/weather-forecast",
                    "currency_forecast": "/api/v1/pipeline/currency-forecast"
                }
            },
            "documentation": "See README.md for detailed API documentation"
        })
    
    # Error handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": "Bad request", "message": str(error)}), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not found", "message": str(error)}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error", "message": str(error)}), 500
    
    return app


if __name__ == "__main__":
    app = create_app()
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║            Time Series Forecasting API v1.0.0                     ║
╠═══════════════════════════════════════════════════════════════════╣
║  Server running on: http://{settings.HOST}:{settings.PORT}                         ║
║  Debug mode: {str(settings.DEBUG).lower()}                                              ║
╠═══════════════════════════════════════════════════════════════════╣
║  Data Sources:                                                    ║
║    - Open-Meteo API (weather data, no auth required)              ║
║    - Frankfurter API (currency rates, no auth required)           ║
╠═══════════════════════════════════════════════════════════════════╣
║  Forecasting Models:                                              ║
║    POST /api/v1/forecast/arima             - ARIMA model          ║
║    POST /api/v1/forecast/ets               - Exponential Smoothing║
║    POST /api/v1/forecast/moving-average    - Moving Average       ║
║    POST /api/v1/forecast/linear-regression - Linear Regression    ║
║    POST /api/v1/forecast/neural-network    - Neural Network       ║
╠═══════════════════════════════════════════════════════════════════╣
║  Quick Start Pipelines:                                           ║
║    GET /api/v1/pipeline/weather-forecast                          ║
║    GET /api/v1/pipeline/currency-forecast                         ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    app.run(
        host=settings.HOST,
        port=settings.PORT,
        debug=settings.DEBUG
    )
