# Indonesian Stock Price Forecasting with MLflow

This project implements a comprehensive machine learning pipeline for forecasting Indonesian stock prices using various regression models and MLflow for experiment tracking.

## 🎯 Objective

Develop a machine learning model to forecast the closing price of Indonesian stocks for the next trading day, demonstrating MLOps principles through MLflow integration.

## 📊 Features

- **Data Collection**: Automated data retrieval using `yfinance`
- **Feature Engineering**: 50+ technical indicators and lag features
- **Multiple Models**: Linear Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Baseline Comparison**: Naive and moving average baselines
- **MLflow Integration**: Complete experiment tracking and model management
- **Automated Model Selection**: Best model selection based on RMSE

## 🚀 Quick Start

## 🚀 Quick Start (Mac Users)

### Option 1: Automated Setup (Recommended for Mac)

```bash
# Make the setup script executable
chmod +x mac_setup.sh

# Run the automated setup
./mac_setup.sh

# Run the pipeline
python3 stock_forecasting.py
```

### Option 2: Manual Installation

```bash
# Install core dependencies first
pip3 install yfinance pandas numpy scikit-learn mlflow matplotlib seaborn requests

# Try to install optional packages (these might fail, but the script will still work)
pip3 install xgboost  # Optional
pip3 install lightgbm  # Optional

# For TA-Lib (optional):
brew install ta-lib
pip3 install TA-Lib
```

### 1. Installation

```bash
# Clone or download the project files
# Install dependencies
pip install -r requirements.txt

# For TA-Lib on Windows, you might need to install from wheel:
# pip install TA-Lib-0.4.28-cp39-cp39-win_amd64.whl

# For TA-Lib on Ubuntu/Linux:
# sudo apt-get install libta-lib-dev
```

### 2. Run the Pipeline

```bash
# The script will automatically detect available packages and adapt
python3 stock_forecasting.py
```

### 3. View Results in MLflow

```bash
# Start MLflow UI
mlflow ui

# Open browser and navigate to:
# http://localhost:5000
```

## 📁 Project Structure

```
├── stock_forecasting.py      # Main pipeline script
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── data_exploration.png     # Generated data visualization
└── mlruns/                  # MLflow experiments directory
```

## 🔧 Configuration

### Stock Selection
The default stock is Bank Central Asia (BBCA.JK). You can change it by modifying the `STOCK_SYMBOL` variable in the `main()` function:

```python
STOCK_SYMBOL = "ASII.JK"  # Astra International
# or
STOCK_SYMBOL = "TLKM.JK"  # Telkom Indonesia
# or
STOCK_SYMBOL = "UNTR.JK"  # United Tractors
```

### Data Period
Modify the data collection period:

```python
forecaster.load_data(period="5y")  # 5 years of data
```

## 🎛️ Model Configuration

The pipeline includes five regression models:

1. **Linear Regression**: Simple baseline with feature scaling
2. **Random Forest**: Ensemble method with 100 trees
3. **Gradient Boosting**: Sequential ensemble learning
4. **XGBoost**: Optimized gradient boosting
5. **LightGBM**: Fast gradient boosting framework

## 📈 Features Generated

### Price-based Features
- High-Low percentage
- Price change (returns)
- Moving averages (5, 10, 20, 50 days)
- Moving average ratios

### Technical Indicators (using TA-Lib)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator

### Lag Features
- Price lags (1, 2, 3, 5, 10 days)
- Volume lags
- Return lags

### Rolling Statistics
- Rolling max/min prices
- Rolling volume averages
- Volatility measures

## 📊 Model Evaluation

### Metrics Used
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

### Baseline Models
- **Naive Forecast**: Previous day's price
- **Moving Average**: 5-day moving average

## 🔬 MLflow Integration

### Experiment Tracking
- Model parameters and hyperparameters
- Performance metrics (RMSE, MAE, R²)
- Baseline comparisons
- Improvement percentages

### Model Management
- Automatic model logging
- Best model registration
- Artifact storage (plots, model weights)

### Reproducibility
- Random seed management
- Parameter logging
- Environment tracking

## 📋 Output

The pipeline generates:

1. **Data Exploration Plot**: Price trends, volume, distributions
2. **Model Predictions Plots**: Actual vs predicted comparisons
3. **Console Report**: Comprehensive performance summary
4. **MLflow Experiments**: Detailed tracking in web UI

## 🎯 Expected Results

- **Training Time**: 2-5 minutes depending on data size
- **Model Performance**: Typically R² > 0.8 for tree-based models
- **Forecast Accuracy**: Usually within 2-5% of actual next-day price

## 🚨 Troubleshooting

## 🚨 Troubleshooting (Mac-Specific)

### Common Mac Issues

1. **XGBoost/LightGBM Installation Issues**:
   ```bash
   # If you get compilation errors, try using conda instead:
   conda install -c conda-forge xgboost lightgbm
   
   # Or use pre-compiled wheels:
   pip3 install --only-binary=all xgboost lightgbm
   
   # If all else fails, the script works fine without them!
   ```

2. **TA-Lib Installation on Mac**:
   ```bash
   # Install TA-Lib system library first
   brew install ta-lib
   
   # Then install Python wrapper
   pip3 install TA-Lib
   
   # If it fails with M1/M2 Mac:
   arch -arm64 brew install ta-lib
   arch -arm64 pip3 install TA-Lib
   ```

3. **Python Version Issues**:
   ```bash
   # Make sure you're using Python 3.8+
   python3 --version
   
   # If you have multiple Python versions:
   python3.9 -m pip install -r requirements.txt
   python3.9 stock_forecasting.py
   ```

4. **M1/M2 Mac Specific Issues**:
   ```bash
   # If you get architecture errors:
   arch -arm64 brew install python@3.9
   arch -arm64 pip3 install -r requirements.txt
   ```

### Common Issues

1. **TA-Lib Installation Error**:
   ```bash
   # Windows: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
   pip install TA_Lib-0.4.28-cp39-cp39-win_amd64.whl
   
   # Linux/Mac:
   sudo apt-get install libta-lib-dev  # Ubuntu
   brew install ta-lib  # macOS
   ```

2. **yfinance Data Issues**:
   - Check internet connection
   - Verify stock symbol format (e.g., "BBCA.JK" for Indonesian stocks)
   - Try different time periods if data is sparse

3. **MLflow UI Not Loading**:
   ```bash
   # Kill existing MLflow processes
   pkill -f mlflow
   
   # Restart MLflow UI
   mlflow ui --host 0.0.0.0 --port 5000
   ```

## 📊 Sample Results

```
=== STOCK PRICE FORECASTING REPORT - BBCA.JK ===

Data Summary:
- Period: 2022-01-01 to 2024-01-25
- Total samples: 523
- Features used: 45

Model Performance:
- LINEAR_REGRESSION: RMSE: 245.67, MAE: 189.34, R²: 0.823
- RANDOM_FOREST: RMSE: 198.45, MAE: 145.23, R²: 0.887
- XGBOOST: RMSE: 186.23, MAE: 134.56, R²: 0.901

Best Model: XGBOOST
Current price: 8875.00
Predicted next day price: 8920.50
Predicted change: +45.50 (+0.51%)
```

## 🏆 Advanced Usage

### Custom Feature Engineering
```python
# Add custom features in the engineer_features method
df['Custom_Indicator'] = your_custom_calculation(df)
```

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
```

## 📝 License

This project is for educational and research purposes. Please ensure compliance with your local financial regulations when using stock market data.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues:
- Check the troubleshooting section
- Review MLflow documentation: https://mlflow.org/docs/latest/index.html
- Check yfinance documentation: https://pypi.org/project/yfinance/

---

**Note**: This is a technical demonstration. Stock price prediction involves significant risk, and this model should not be used for actual trading decisions without proper validation and risk management.