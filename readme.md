# Indonesian Stock Price Forecasting with MLflow

This project implements a comprehensive machine learning pipeline for forecasting Indonesian stock prices using various regression models and MLflow for experiment tracking.

## 1. Goals

Develop a machine learning model to forecast the closing price of Indonesian stocks for the next trading day, demonstrating MLOps principles through MLflow integration.

## 2. Features

- **Data Collection**: Data retrieval using `yfinance`
- **Feature Engineering**: Additional technical indicators (Comprehensive list on: **6.1 Features Generated**).
- **Multiple Models**: Linear Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Baseline Comparison**: We use moving average as baselines
- **MLflow Integration**: Complete experiment tracking and model management
- **Automated Model Selection**: Best model selection based on RMSE

## 3. Quick Start

### 3.1 Installation

```bash
# Clone the repo using
git clone git@github.com:izzako/stock-prices-forecasting.git

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install yfinance pandas numpy scikit-learn mlflow matplotlib seaborn requests

# Try to install optional packages (these might fail, but the script will still work)
pip install xgboost  # Optional
pip install lightgbm  # Optional

# For TA-Lib on Windows, you might need to install from wheel:
# pip install TA-Lib-0.4.28-cp39-cp39-win_amd64.whl

# For TA-Lib on Ubuntu/Linux:
# sudo apt-get install libta-lib-dev

# For TA-Lib on Mac:
# brew install ta-lib

pip install TA-Lib

# or using requirements.txt
# pip install -r requirements.txt
```

### 3.2 Run the Pipeline

```bash
# The script will automatically detect available packages and adapt
python main.py
```

### 3.3 View Results in MLflow

```bash
# Start MLflow UI on custom port (8080)
mlflow server --host 127.0.0.1 --port 8080

# Open browser and navigate to:
# http://localhost:8080
```

## 4. Project Structure

```
├── main.py      # Main pipeline script
├── venv/        # virtual environment
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── data_exploration.png     # Generated data exploration viz
├── regressor_image          # Generated Test Data Prediction Viz
└── mlruns/                  # MLflow experiments directory
```

## 5. Config
### 5.1 Stock Selection

The default stock is Bank Central Asia (`BBCA.JK`). You can change it by modifying the `STOCK_SYMBOL` variable in the `run_regression ()` function:

```python
STOCK_SYMBOL = "ASII.JK"  # Astra International
# or
STOCK_SYMBOL = "TLKM.JK"  # Telkom Indonesia
# or
STOCK_SYMBOL = "UNTR.JK"  # United Tractors
```

### 5.2 Data Period
Modify the data collection period:

```python
forecaster.load_data(period="5y")  # 5 years of data
```

## 6. Model Options

The pipeline includes five regression models:

1. **Linear Regression**: Simple baseline with feature scaling
2. **Random Forest**: Ensemble method with 200 trees
3. **Gradient Boosting**: Sequential ensemble learning
4. **XGBoost**: Optimized gradient boosting
5. **LightGBM**: Fast gradient boosting framework

## 7. Features Generated

### 7.1 Price-based Features
- High-Low percentage
- Price change (returns)
- Moving averages (5, 10, 20, 50 days)
- Moving average ratios

### 7.2 Technical Indicators (using TA-Lib)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator

### 7.3 Lag Features
- Price lags (1, 5, 10 days)
- Volume lags (1, 5, 10 days)
- Return lags (1, 5, 10 days)

### 7.4 Rolling Statistics
- Rolling max/min prices (5, 10, 20 days)
- Rolling volume averages (5, 10, 20 days)
- Volatility measures (5, 10, 20 days)

## 8. Model Evaluation

### 8.1 Metrics Used
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

### 8.2 Baseline Models
- **Moving Average**: 5-day moving average

## 9. MLflow Integration

### 9.1 Experiment Tracking
- Model parameters and hyperparameters
- Performance metrics (RMSE, MAE, R²)
- Baseline comparisons
- Improvement percentages

### 9.2 Model Management
- Automatic model logging
- Best model registration
- Artifact storage (plots, model weights)

### 9.3 Reproducibility
- Random seed management
- Parameter logging
- Environment tracking

## 10. Output

The pipeline generates:

1. **Data Exploration Plot**: Price trends, volume, distributions
2. **Model Predictions Plots**: Actual vs predicted comparisons
3. **Console Report**: Comprehensive performance summary
4. **MLflow Experiments**: Detailed tracking in web UI

## 11. Expected Results

- **Training Time**: 2-5 minutes depending on data size
- **Model Performance**: Almost 0.7-0.8 R² performances for tree-based models
- **Forecast Accuracy**: Usually within 2-5% of actual next-day price



## 12. Results

Full results are in `sample_run_results/console_report.md`.

```
============================================================
STOCK PRICE FORECASTING REPORT - BBCA.JK
============================================================

Data Summary:
- Period: 2020-07-29 to 2025-07-29
- Total samples: 1200
- Features used: 42

Model Performance:
- LINEAR_REGRESSION:
  * RMSE: 164.6530
  * MAE: 125.8647
  * R²: 0.8146
- RANDOM_FOREST:
  * RMSE: 213.5445
  * MAE: 168.3006
  * R²: 0.6881
- GRADIENT_BOOSTING:
  * RMSE: 207.0558
  * MAE: 161.6780
  * R²: 0.7068
- XGBOOST:
  * RMSE: 227.3087
  * MAE: 180.3070
  * R²: 0.6466
- LIGHTGBM:
  * RMSE: 213.5368
  * MAE: 166.5430
  * R²: 0.6881

Best Model: LINEAR_REGRESSION
RMSE: 164.6530

=== Forecast Results ===
Model used: linear_regression
Current price at 2025-07-29: 8400.00
Predicted next day price at 2025-07-30: 8474.33
Predicted change: 74.33 (0.88%)
```

MLFlow Snapshot
![mlflow snapshot](sample_run_results/mlflow_snapshot.png)

## 13. Advanced Usage

**Custom Feature Engineering**
```python
# Add custom features in the engineer_features method
df['Custom_Indicator'] = your_custom_calculation(df)
```

<!-- 
### Hyperparameter Tuning
```python
# Add GridSearchCV or RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
```

### Multiple Stock Analysis
```python
stocks = ['BBCA.JK', 'ASII.JK', 'TLKM.JK']
for stock in stocks:
    forecaster = StockPriceForecaster(stock)
    # Run pipeline for each stock
``` -->
