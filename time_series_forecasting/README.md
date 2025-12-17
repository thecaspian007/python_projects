# Time Series Forecasting API

A Flask-based REST API for time series data fetching, forecasting, and evaluation. This API integrates with free external data sources and provides multiple statistical and machine learning models for forecasting.

## Features

- **Live Data Fetching**: Weather data from Open-Meteo, Currency rates from Frankfurter
- **Statistical Models**: ARIMA, ETS (Exponential Smoothing), Moving Averages
- **ML Models**: Linear/Polynomial Regression, Neural Networks
- **Comprehensive Metrics**: RMSE, MAE, MAPE, SMAPE, R²

## Installation

### Prerequisites

- Python 3.9+
- pip package manager

### Setup

1. **Navigate to directory**:
   ```bash
   cd time_series_forecasting
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

The API will start on `http://localhost:5001`

## API Endpoints

### Health Check
```
GET /api/v1/health
```

---

## Data Fetching Endpoints

### Get Weather Data
```
GET /api/v1/datasets/weather
```

Fetch historical weather data from Open-Meteo API (no authentication required).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| location | String | new_york | Location name |
| days | Integer | 30 | Days of historical data |
| variable | String | temperature_2m | Weather variable |

**Available Locations**: new_york, london, tokyo, paris, sydney, mumbai

**Example**:
```bash
curl "http://localhost:5001/api/v1/datasets/weather?location=london&days=14"
```

### Get Currency Data
```
GET /api/v1/datasets/currency
```

Fetch historical exchange rates from Frankfurter API (no authentication required).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| base | String | EUR | Base currency code |
| target | String | USD | Target currency code |
| days | Integer | 30 | Days of historical data |

**Example**:
```bash
curl "http://localhost:5001/api/v1/datasets/currency?base=GBP&target=INR&days=30"
```

---

## Forecasting Endpoints

All forecasting endpoints accept POST requests with JSON body.

### ARIMA Forecast
```
POST /api/v1/forecast/arima
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| values | Array | required | Time series values |
| timestamps | Array | auto | Timestamps for values |
| forecast_steps | Integer | 10 | Steps to forecast |
| order | Array | [5,1,0] | ARIMA order [p,d,q] |

**Example**:
```bash
curl -X POST http://localhost:5001/api/v1/forecast/arima \
  -H "Content-Type: application/json" \
  -d '{"values": [10, 12, 14, 13, 15, 17, 16, 18, 20, 19], "forecast_steps": 5}'
```

### ETS (Exponential Smoothing) Forecast
```
POST /api/v1/forecast/ets
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| values | Array | required | Time series values |
| forecast_steps | Integer | 10 | Steps to forecast |
| trend | String | add | Trend: 'add', 'mul', or null |
| seasonal | String | null | Seasonal: 'add', 'mul', or null |
| seasonal_periods | Integer | null | Length of seasonal cycle |

### Moving Average Forecast
```
POST /api/v1/forecast/moving-average
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| values | Array | required | Time series values |
| forecast_steps | Integer | 10 | Steps to forecast |
| window_size | Integer | 5 | MA window size |
| method | String | sma | 'sma', 'wma', or 'ema' |

### Linear Regression Forecast
```
POST /api/v1/forecast/linear-regression
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| values | Array | required | Time series values |
| forecast_steps | Integer | 10 | Steps to forecast |
| degree | Integer | 1 | Polynomial degree |

### Neural Network Forecast
```
POST /api/v1/forecast/neural-network
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| values | Array | required | Time series values |
| forecast_steps | Integer | 10 | Steps to forecast |
| sequence_length | Integer | 10 | Input sequence length |
| epochs | Integer | 50 | Training epochs |

---

## Pipeline Endpoints (Quick Start)

Complete data fetching + forecasting in one call.

### Weather Forecast Pipeline
```
GET /api/v1/pipeline/weather-forecast
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| location | String | new_york | Location name |
| days | Integer | 30 | Historical days |
| forecast_steps | Integer | 7 | Steps to forecast |
| model | String | arima | Model: arima, ets, ma, lr |

**Example**:
```bash
curl "http://localhost:5001/api/v1/pipeline/weather-forecast?location=tokyo&model=ets&forecast_steps=7"
```

### Currency Forecast Pipeline
```
GET /api/v1/pipeline/currency-forecast
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| base | String | EUR | Base currency |
| target | String | USD | Target currency |
| days | Integer | 30 | Historical days |
| forecast_steps | Integer | 7 | Steps to forecast |
| model | String | arima | Model to use |

---

## Evaluation Endpoints

### Evaluate Forecast
```
POST /api/v1/evaluate
```

| Parameter | Type | Description |
|-----------|------|-------------|
| actual | Array | Actual values |
| predicted | Array | Predicted values |

**Example**:
```bash
curl -X POST http://localhost:5001/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"actual": [10, 12, 14, 13, 15], "predicted": [11, 13, 13, 14, 14]}'
```

### Compare Models
```
POST /api/v1/evaluate/compare
```

Compare multiple model predictions.

---

## Statistical Models Used

### 1. ARIMA (AutoRegressive Integrated Moving Average)

| Component | Description |
|-----------|-------------|
| **AR (p)** | Uses past values to predict future |
| **I (d)** | Differencing for stationarity |
| **MA (q)** | Uses past forecast errors |

**Best for**: Data with trends and autocorrelation

### 2. ETS (Error, Trend, Seasonal)

| Method | Description |
|--------|-------------|
| **Simple** | No trend, no seasonality |
| **Holt's Linear** | Additive trend |
| **Holt-Winters** | Trend + seasonality |

**Best for**: Data with clear trend and/or seasonal patterns

### 3. Moving Average

| Type | Description |
|------|-------------|
| **SMA** | Simple - equal weights |
| **WMA** | Weighted - linear weights |
| **EMA** | Exponential - exponential decay |

**Best for**: Short-term forecasting, trend smoothing

### 4. Linear Regression

Fits a linear (or polynomial) function to model the trend.

| Degree | Description |
|--------|-------------|
| 1 | Linear trend |
| 2 | Quadratic trend |
| 3+ | Higher-order polynomial |

**Best for**: Data with clear linear trends

### 5. Neural Network

Feedforward neural network with configurable architecture.

| Feature | Description |
|---------|-------------|
| Input | Sequence of past values |
| Hidden | Configurable dense layers |
| Output | Next value prediction |

**Best for**: Complex non-linear patterns

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | √(mean(errors²)) | Lower is better, penalizes large errors |
| **MAE** | mean(\|errors\|) | Lower is better, more robust to outliers |
| **MAPE** | mean(\|errors/actual\|) × 100 | Percentage error |
| **SMAPE** | Symmetric MAPE | Handles zeros better |
| **R²** | 1 - (SS_res/SS_tot) | 1.0 = perfect, 0 = mean baseline |

---

## Project Structure

```
time_series_forecasting/
├── app.py                      # Flask application
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── config/
│   └── settings.py             # Configuration
├── controllers/
│   └── forecast_controller.py  # REST API endpoints
├── services/
│   ├── base_analyzer.py        # Abstract base class
│   ├── statistical/
│   │   ├── arima_service.py
│   │   ├── ets_service.py
│   │   └── moving_average_service.py
│   ├── ml_based/
│   │   ├── linear_regression_service.py
│   │   └── neural_network_service.py
│   └── evaluation/
│       └── metrics_service.py
├── models/
│   ├── forecast_result.py
│   └── time_series_data.py
├── utils/
│   └── data_utils.py
└── external_apis/
    ├── base_client.py
    ├── open_meteo_client.py
    └── frankfurter_client.py
```

## External APIs Used

### Open-Meteo
- **URL**: https://open-meteo.com/
- **Authentication**: None required
- **Rate Limit**: 10,000 requests/day
- **Data**: Historical weather data worldwide

### Frankfurter
- **URL**: https://frankfurter.app/
- **Authentication**: None required
- **Data Source**: European Central Bank
- **Data**: Historical exchange rates

## License

MIT License
