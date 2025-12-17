"""
Time Series Forecasting API Controller.

This module contains all REST API endpoints for time series data fetching,
forecasting, and evaluation operations.
"""
from flask import Blueprint, request, jsonify
import traceback

from config.settings import settings
from models.time_series_data import TimeSeriesData
from external_apis.open_meteo_client import OpenMeteoClient
from external_apis.frankfurter_client import FrankfurterClient
from services.statistical.arima_service import ARIMAService
from services.statistical.ets_service import ETSService
from services.statistical.moving_average_service import MovingAverageService
from services.ml_based.linear_regression_service import LinearRegressionService
from services.ml_based.neural_network_service import NeuralNetworkService
from services.evaluation.metrics_service import MetricsService


# Create Blueprint
forecast_bp = Blueprint("forecast", __name__, url_prefix="/api/v1")

# Initialize clients
open_meteo_client = OpenMeteoClient()
frankfurter_client = FrankfurterClient()

# Initialize services
arima_service = ARIMAService()
ets_service = ETSService()
ma_service = MovingAverageService()
lr_service = LinearRegressionService()
nn_service = NeuralNetworkService()


@forecast_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Time Series Forecasting API",
        "version": "1.0.0"
    })


# ==================== Dataset Endpoints ====================

@forecast_bp.route("/datasets/weather", methods=["GET"])
def get_weather_data():
    """
    Fetch weather time series data from Open-Meteo API.
    
    Query Parameters:
        location: Location name (new_york, london, tokyo, paris, sydney, mumbai)
        days: Number of days of historical data (default: 30)
        variable: Weather variable (temperature_2m, precipitation, etc.)
    
    Returns:
        JSON with time series data
    """
    try:
        location = request.args.get("location", "new_york")
        days = int(request.args.get("days", 30))
        variable = request.args.get("variable", "temperature_2m")
        
        if variable == "temperature_2m":
            data = open_meteo_client.get_temperature_series(location=location, days=days)
        elif variable == "precipitation":
            data = open_meteo_client.get_precipitation_series(location=location, days=days)
        else:
            data = open_meteo_client.get_temperature_series(location=location, days=days)
        
        return jsonify(data.to_dict())
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@forecast_bp.route("/datasets/currency", methods=["GET"])
def get_currency_data():
    """
    Fetch currency exchange rate time series from Frankfurter API.
    
    Query Parameters:
        base: Base currency (default: EUR)
        target: Target currency (default: USD)
        days: Number of days of historical data (default: 30)
    
    Returns:
        JSON with time series data
    """
    try:
        base = request.args.get("base", "EUR")
        target = request.args.get("target", "USD")
        days = int(request.args.get("days", 30))
        
        data = frankfurter_client.get_exchange_rate_series(
            base=base,
            target=target,
            days=days
        )
        
        return jsonify(data.to_dict())
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@forecast_bp.route("/datasets/locations", methods=["GET"])
def get_available_locations():
    """Get list of available weather locations."""
    return jsonify({
        "locations": open_meteo_client.get_available_locations(),
        "variables": open_meteo_client.get_available_variables()
    })


@forecast_bp.route("/datasets/currencies", methods=["GET"])
def get_available_currencies():
    """Get list of available currencies."""
    try:
        currencies = frankfurter_client.get_available_currencies()
        return jsonify({"currencies": currencies})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Forecasting Endpoints ====================

def _get_data_from_request():
    """Extract time series data from request body."""
    if not request.is_json:
        return None, "Request must be JSON"
    
    data = request.get_json()
    
    if "values" not in data:
        return None, "Missing 'values' in request body"
    
    try:
        ts_data = TimeSeriesData(
            name=data.get("name", "input_series"),
            values=data["values"],
            timestamps=data.get("timestamps", [str(i) for i in range(len(data["values"]))]),
            source="user_input"
        )
        return ts_data, None
    except Exception as e:
        return None, f"Invalid data format: {str(e)}"


@forecast_bp.route("/forecast/arima", methods=["POST"])
def forecast_arima():
    """
    Generate forecast using ARIMA model.
    
    Request Body (JSON):
        values: List of time series values
        timestamps: Optional list of timestamps
        forecast_steps: Number of steps to forecast (default: 10)
        order: Optional ARIMA order [p, d, q]
    
    Returns:
        JSON with forecast results
    """
    data, error = _get_data_from_request()
    if error:
        return jsonify({"error": error}), 400
    
    try:
        body = request.get_json()
        steps = body.get("forecast_steps", 10)
        order = tuple(body.get("order", settings.ARIMA_DEFAULT_ORDER))
        
        arima_service.fit(data, order=order)
        result = arima_service.predict(steps)
        
        return jsonify(result.to_dict())
        
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@forecast_bp.route("/forecast/ets", methods=["POST"])
def forecast_ets():
    """
    Generate forecast using Exponential Smoothing (ETS) model.
    
    Request Body (JSON):
        values: List of time series values
        timestamps: Optional list of timestamps
        forecast_steps: Number of steps to forecast (default: 10)
        trend: Trend component ('add', 'mul', or null)
        seasonal: Seasonal component ('add', 'mul', or null)
        seasonal_periods: Number of periods in seasonal cycle
    
    Returns:
        JSON with forecast results
    """
    data, error = _get_data_from_request()
    if error:
        return jsonify({"error": error}), 400
    
    try:
        body = request.get_json()
        steps = body.get("forecast_steps", 10)
        trend = body.get("trend", "add")
        seasonal = body.get("seasonal", None)
        seasonal_periods = body.get("seasonal_periods", None)
        
        ets_service.fit(
            data,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
        result = ets_service.predict(steps)
        
        return jsonify(result.to_dict())
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@forecast_bp.route("/forecast/moving-average", methods=["POST"])
def forecast_moving_average():
    """
    Generate forecast using Moving Average.
    
    Request Body (JSON):
        values: List of time series values
        forecast_steps: Number of steps to forecast (default: 10)
        window_size: Size of moving average window (default: 5)
        method: Type of MA (sma, wma, ema)
    
    Returns:
        JSON with forecast results
    """
    data, error = _get_data_from_request()
    if error:
        return jsonify({"error": error}), 400
    
    try:
        body = request.get_json()
        steps = body.get("forecast_steps", 10)
        window_size = body.get("window_size", 5)
        method = body.get("method", "sma")
        
        ma_service.fit(data, window_size=window_size, method=method)
        result = ma_service.predict(steps)
        
        return jsonify(result.to_dict())
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@forecast_bp.route("/forecast/linear-regression", methods=["POST"])
def forecast_linear_regression():
    """
    Generate forecast using Linear Regression.
    
    Request Body (JSON):
        values: List of time series values
        forecast_steps: Number of steps to forecast (default: 10)
        degree: Polynomial degree (default: 1)
    
    Returns:
        JSON with forecast results
    """
    data, error = _get_data_from_request()
    if error:
        return jsonify({"error": error}), 400
    
    try:
        body = request.get_json()
        steps = body.get("forecast_steps", 10)
        degree = body.get("degree", 1)
        
        lr_service.fit(data, degree=degree)
        result = lr_service.predict(steps)
        
        return jsonify(result.to_dict())
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@forecast_bp.route("/forecast/neural-network", methods=["POST"])
def forecast_neural_network():
    """
    Generate forecast using Neural Network.
    
    Request Body (JSON):
        values: List of time series values
        forecast_steps: Number of steps to forecast (default: 10)
        sequence_length: Input sequence length (default: 10)
        epochs: Training epochs (default: 50)
    
    Returns:
        JSON with forecast results
    """
    data, error = _get_data_from_request()
    if error:
        return jsonify({"error": error}), 400
    
    try:
        body = request.get_json()
        steps = body.get("forecast_steps", 10)
        sequence_length = body.get("sequence_length", 10)
        epochs = body.get("epochs", 50)
        
        nn_service.fit(data, sequence_length=sequence_length, epochs=epochs)
        result = nn_service.predict(steps)
        
        return jsonify(result.to_dict())
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== Evaluation Endpoints ====================

@forecast_bp.route("/evaluate", methods=["POST"])
def evaluate_forecast():
    """
    Evaluate forecast accuracy.
    
    Request Body (JSON):
        actual: List of actual values
        predicted: List of predicted values
        model_name: Optional model name
    
    Returns:
        JSON with evaluation metrics
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    body = request.get_json()
    
    if "actual" not in body or "predicted" not in body:
        return jsonify({"error": "Missing 'actual' or 'predicted' values"}), 400
    
    try:
        actual = body["actual"]
        predicted = body["predicted"]
        model_name = body.get("model_name", "unknown")
        
        result = MetricsService.evaluate(model_name, actual, predicted)
        
        return jsonify(result.to_dict())
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@forecast_bp.route("/evaluate/compare", methods=["POST"])
def compare_models():
    """
    Compare multiple models on the same data.
    
    Request Body (JSON):
        actual: List of actual values
        predictions: Dict mapping model names to predicted values
    
    Returns:
        JSON with comparison metrics for each model
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    body = request.get_json()
    
    if "actual" not in body or "predictions" not in body:
        return jsonify({"error": "Missing 'actual' or 'predictions'"}), 400
    
    try:
        comparison = MetricsService.compare_models(
            body["actual"],
            body["predictions"]
        )
        
        return jsonify({
            "comparison": comparison,
            "best_model": {
                metric: min(comparison.keys(), key=lambda m: comparison[m][metric])
                for metric in ["rmse", "mae", "mape"]
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@forecast_bp.route("/metrics", methods=["GET"])
def get_metrics_info():
    """Get information about available metrics."""
    return jsonify({
        "metrics": MetricsService.get_metric_descriptions()
    })


# ==================== Full Pipeline Endpoint ====================

@forecast_bp.route("/pipeline/weather-forecast", methods=["GET"])
def weather_forecast_pipeline():
    """
    Complete pipeline: Fetch weather data and generate forecasts.
    
    Query Parameters:
        location: Location name (default: new_york)
        days: Historical days (default: 30)
        forecast_steps: Steps to forecast (default: 7)
        model: Model to use (arima, ets, ma, lr) (default: arima)
    
    Returns:
        JSON with data and forecast
    """
    try:
        location = request.args.get("location", "new_york")
        days = int(request.args.get("days", 30))
        steps = int(request.args.get("forecast_steps", 7))
        model = request.args.get("model", "arima")
        
        # Fetch data
        data = open_meteo_client.get_temperature_series(location=location, days=days)
        
        # Generate forecast based on model choice
        if model == "arima":
            arima_service.fit(data)
            result = arima_service.predict(steps)
        elif model == "ets":
            ets_service.fit(data)
            result = ets_service.predict(steps)
        elif model == "ma":
            ma_service.fit(data)
            result = ma_service.predict(steps)
        elif model == "lr":
            lr_service.fit(data)
            result = lr_service.predict(steps)
        else:
            return jsonify({"error": f"Unknown model: {model}"}), 400
        
        return jsonify({
            "data": data.to_dict(),
            "forecast": result.to_dict()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@forecast_bp.route("/pipeline/currency-forecast", methods=["GET"])
def currency_forecast_pipeline():
    """
    Complete pipeline: Fetch currency data and generate forecasts.
    
    Query Parameters:
        base: Base currency (default: EUR)
        target: Target currency (default: USD)
        days: Historical days (default: 30)
        forecast_steps: Steps to forecast (default: 7)
        model: Model to use (arima, ets, ma, lr) (default: arima)
    
    Returns:
        JSON with data and forecast
    """
    try:
        base = request.args.get("base", "EUR")
        target = request.args.get("target", "USD")
        days = int(request.args.get("days", 30))
        steps = int(request.args.get("forecast_steps", 7))
        model = request.args.get("model", "arima")
        
        # Fetch data
        data = frankfurter_client.get_exchange_rate_series(
            base=base,
            target=target,
            days=days
        )
        
        # Generate forecast
        if model == "arima":
            arima_service.fit(data)
            result = arima_service.predict(steps)
        elif model == "ets":
            ets_service.fit(data)
            result = ets_service.predict(steps)
        elif model == "ma":
            ma_service.fit(data)
            result = ma_service.predict(steps)
        elif model == "lr":
            lr_service.fit(data)
            result = lr_service.predict(steps)
        else:
            return jsonify({"error": f"Unknown model: {model}"}), 400
        
        return jsonify({
            "data": data.to_dict(),
            "forecast": result.to_dict()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
