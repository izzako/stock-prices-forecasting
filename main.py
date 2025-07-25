import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Optional libraries - handle import errors gracefully
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    print("LightGBM not available. Install with: pip install lightgbm")
    LGB_AVAILABLE = False

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Optional MLflow integrations
if XGB_AVAILABLE:
    import mlflow.xgboost
if LGB_AVAILABLE:
    import mlflow.lightgbm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')

# Technical Analysis - handle TA-Lib import gracefully
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    print("TA-Lib not available. Install with: brew install ta-lib && pip install TA-Lib")
    TALIB_AVAILABLE = False

class StockPriceRegressor:
    """
    A comprehensive stock price regression system with MLflow integration.
    
    This class handles data loading, preprocessing, feature engineering,
    regression model training, evaluation, and MLflow experiment tracking.
    """
    
    def __init__(self, stock_symbol, experiment_name="stock_regression"):
        """
        Initialize the regressor.
        
        Args:
            stock_symbol (str): Stock ticker symbol (e.g., 'BBCA.JK')
            experiment_name (str): MLflow experiment name
        """
        self.stock_symbol = stock_symbol
        self.experiment_name = experiment_name
        self.data = None
        self.features = None
        self.target = None
        self.scaler = StandardScaler()
        self.models = {}
        
        # Initialize MLflow
        mlflow.set_experiment(self.experiment_name)
        
    def load_data(self, period="2y", interval="1d"):
        """
        Load stock data using yfinance.
        
        Args:
            period (str): Data period (e.g., '2y', '5y')
            interval (str): Data interval (e.g., '1d', '1h')
        """
        try:
            print(f"Loading data for {self.stock_symbol}...")
            ticker = yf.Ticker(self.stock_symbol)
            self.data = ticker.history(period=period, interval=interval)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.stock_symbol}")
                
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
            
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
            
    def explore_data(self):
        """Perform basic data exploration and visualization."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        print("\n=== Data Exploration ===")
        print(f"Data shape: {self.data.shape}")
        print(f"Missing values:\n{self.data.isnull().sum()}")
        print(f"\nData types:\n{self.data.dtypes}")
        print(f"\nBasic statistics:\n{self.data.describe()}")
        
        # Plot price data
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Closing price over time
        axes[0, 0].plot(self.data.index, self.data['Close'])
        axes[0, 0].set_title(f'{self.stock_symbol} - Closing Price')
        axes[0, 0].set_ylabel('Price')
        
        # Volume over time
        axes[0, 1].plot(self.data.index, self.data['Volume'])
        axes[0, 1].set_title(f'{self.stock_symbol} - Volume')
        axes[0, 1].set_ylabel('Volume')
        
        # Price distribution
        axes[1, 0].hist(self.data['Close'], bins=50, alpha=0.7)
        axes[1, 0].set_title('Closing Price Distribution')
        axes[1, 0].set_xlabel('Price')
        
        # Daily returns
        daily_returns = self.data['Close'].pct_change().dropna()
        axes[1, 1].hist(daily_returns, bins=50, alpha=0.7)
        axes[1, 1].set_title('Daily Returns Distribution')
        axes[1, 1].set_xlabel('Returns')
        
        plt.tight_layout()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def engineer_features(self):
        """
        Create technical indicators and lag features for model training.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        print("\n=== Feature Engineering ===")
        df = self.data.copy()
        
        # Basic price features
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'].pct_change()
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}']
        
        # Volatility features
        df['Volatility_5'] = df['Close'].rolling(window=5).std()
        df['Volatility_20'] = df['Close'].rolling(window=20).std()
        
        # Technical indicators using TA-Lib (if available)
        if TALIB_AVAILABLE:
            try:
                # RSI
                df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(df['Close'].values)
                df['MACD'] = macd
                df['MACD_Signal'] = macd_signal
                df['MACD_Hist'] = macd_hist
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'].values)
                df['BB_Upper'] = bb_upper
                df['BB_Lower'] = bb_lower
                df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
                df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
                
                # Stochastic Oscillator
                df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'].values, 
                                                          df['Low'].values, 
                                                          df['Close'].values)
                print("TA-Lib indicators added successfully")
            except Exception as e:
                print(f"Error with TA-Lib indicators: {e}")
        else:
            # Manual technical indicators as fallback
            print("Using manual technical indicators (TA-Lib not available)")
            
            # Simple RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Simple MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Simple Bollinger Bands
            bb_middle = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = bb_middle + (bb_std * 2)
            df['BB_Lower'] = bb_middle - (bb_std * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / bb_middle
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Return_Lag_{lag}'] = df['Price_Change'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'Close_Rolling_Max_{window}'] = df['Close'].rolling(window=window).max()
            df[f'Close_Rolling_Min_{window}'] = df['Close'].rolling(window=window).min()
            df[f'Volume_Rolling_Mean_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # Target variable (next day's closing price)
        df['Target'] = df['Close'].shift(-1)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        self.features = df[feature_cols]
        self.target = df['Target']
        
        print(f"Features created: {len(feature_cols)}")
        print(f"Final dataset shape: {df.shape}")
        print(f"Feature columns: {feature_cols[:10]}...")  # Show first 10 features
        
        return self.features, self.target
    
    def create_baseline_model(self, X_test, y_test):
        """
        Create simple baseline models for comparison.
        
        Args:
            X_test: Test features
            y_test: Test target values
        """
        print("\n=== Creating Baseline Models ===")
        
        # Naive forecast (previous day's price)
        naive_pred = X_test['Close_Lag_1'].values
        naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
        naive_mae = mean_absolute_error(y_test, naive_pred)
        naive_r2 = r2_score(y_test, naive_pred)
        
        # Moving average forecast
        ma_pred = X_test['MA_5'].values
        ma_rmse = np.sqrt(mean_squared_error(y_test, ma_pred))
        ma_mae = mean_absolute_error(y_test, ma_pred)
        ma_r2 = r2_score(y_test, ma_pred)
        
        baselines = {
            'naive': {'rmse': naive_rmse, 'mae': naive_mae, 'r2': naive_r2},
            'moving_average': {'rmse': ma_rmse, 'mae': ma_mae, 'r2': ma_r2}
        }
        
        print(f"Naive forecast RMSE: {naive_rmse:.4f}")
        print(f"Moving average RMSE: {ma_rmse:.4f}")
        
        return baselines
    
    def train_models(self, test_size=0.15, random_state=42):
        """
        Train multiple regression models and track experiments with MLflow.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        if self.features is None or self.target is None:
            raise ValueError("Features not engineered. Call engineer_features() first.")
        
        print("\n=== Training Models ===")
        
        # Split data (time series aware)
        split_idx = int(len(self.features) * (1 - test_size))
        X_train = self.features.iloc[:split_idx]
        X_test = self.features.iloc[split_idx:]
        y_train = self.target.iloc[:split_idx]
        y_test = self.target.iloc[split_idx:]
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create baseline models
        baselines = self.create_baseline_model(X_test, y_test)
        
        # Define models to train (conditionally include XGB and LGB)
        models_config = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        }
        
        # Add XGBoost if available
        if XGB_AVAILABLE:
            models_config['xgboost'] = xgb.XGBRegressor(n_estimators=100, random_state=random_state)
        
        # Add LightGBM if available
        if LGB_AVAILABLE:
            models_config['lightgbm'] = lgb.LGBMRegressor(n_estimators=100, random_state=random_state, verbose=-1)
        
        best_model = None
        best_rmse = float('inf')
        
        # Train each model
        for model_name, model in models_config.items():
            with mlflow.start_run(run_name=f"{model_name}_{self.stock_symbol}"):
                print(f"\nTraining {model_name}...")
                
                # Log parameters
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("stock_symbol", self.stock_symbol)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("train_size", len(X_train))
                
                # Use scaled data for linear models, original for tree-based
                if model_name in ['linear_regression']:
                    X_train_model = X_train_scaled
                    X_test_model = X_test_scaled
                else:
                    X_train_model = X_train
                    X_test_model = X_test
                
                # Train model
                model.fit(X_train_model, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_model)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Log metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2_score", r2)
                
                # Log baseline comparison
                mlflow.log_metric("baseline_naive_rmse", baselines['naive']['rmse'])
                mlflow.log_metric("baseline_ma_rmse", baselines['moving_average']['rmse'])
                mlflow.log_metric("improvement_over_naive", (baselines['naive']['rmse'] - rmse) / baselines['naive']['rmse'])
                
                # Log model with appropriate MLflow integration
                if model_name == 'xgboost' and XGB_AVAILABLE:
                    mlflow.xgboost.log_model(model, name="model")
                elif model_name == 'lightgbm' and LGB_AVAILABLE:
                    mlflow.lightgbm.log_model(model, name="model")
                else:
                    mlflow.sklearn.log_model(model, name="model")
                
                # Create and log prediction plot
                os.makedirs('regressor_image', exist_ok=True)
                plt.figure(figsize=(12, 6))
                plt.plot(y_test.values[:100], label='Actual', alpha=0.7)
                plt.plot(y_pred[:100], label='Predicted', alpha=0.7)
                plt.title(f'{model_name} - Actual vs Predicted (First 100 test samples)')
                plt.xlabel('Sample')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plot_path = f'regressor_image/{model_name}_predictions.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(plot_path)
                plt.close()
                
                # Store model and results
                self.models[model_name] = {
                    'model': model,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'predictions': y_pred
                }
                
                # Track best model
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model_name
                
                print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        print(f"\nBest model: {best_model} (RMSE: {best_rmse:.4f})")
        
        # Register best model
        with mlflow.start_run(run_name=f"best_model_{self.stock_symbol}"):
            mlflow.log_param("best_model", best_model)
            mlflow.log_metric("best_rmse", best_rmse)
            
            best_model_obj = self.models[best_model]['model']
            if best_model == 'xgboost' and XGB_AVAILABLE:
                mlflow.xgboost.log_model(best_model_obj, name="best_model", registered_model_name=f"{self.stock_symbol}_regressor")
            elif best_model == 'lightgbm' and LGB_AVAILABLE:
                mlflow.lightgbm.log_model(best_model_obj, name="best_model", registered_model_name=f"{self.stock_symbol}_regressor")
            else:
                mlflow.sklearn.log_model(best_model_obj, name="best_model", registered_model_name=f"{self.stock_symbol}_regressor")
        
        return self.models, best_model
    
    def generate_forecast(self, model_name=None, days_ahead=1):
        """
        Generate forecast for the next trading day(s).
        
        Args:
            model_name (str): Name of the model to use for forecasting
            days_ahead (int): Number of days to forecast
        """
        if not self.models:
            raise ValueError("No models trained. Call train_models() first.")
        
        if model_name is None:
            # Use the best model (lowest RMSE)
            model_name = min(self.models.keys(), key=lambda k: self.models[k]['rmse'])
        
        model = self.models[model_name]['model']
        
        # Get the latest features
        latest_features = self.features.iloc[-1:].copy()
        
        # Scale if needed
        if model_name in ['linear_regression']:
            latest_features_scaled = self.scaler.transform(latest_features)
            forecast = model.predict(latest_features_scaled)[0]
        else:
            forecast = model.predict(latest_features)[0]
        
        current_price = self.data['Close'].iloc[-1]
        today = self.data.iloc[-1].name.strftime('%Y-%m-%d')
        forecast_date = (self.data.index[-1] + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        print(f"\n=== Forecast Results ===")
        print(f"Model used: {model_name}")
        print(f"Current price at {today}: {current_price:.2f}")
        print(f"Predicted next day price at {forecast_date}: {forecast:.2f}")
        print(f"Predicted change: {forecast - current_price:.2f} ({((forecast - current_price) / current_price) * 100:.2f}%)")
        
        return forecast
    
    def create_summary_report(self):
        """Create a comprehensive summary report."""
        if not self.models:
            raise ValueError("No models trained. Call train_models() first.")
        
        print("\n" + "="*60)
        print(f"STOCK PRICE FORECASTING REPORT - {self.stock_symbol}")
        print("="*60)
        
        print(f"\nData Summary:")
        print(f"- Period: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        print(f"- Total samples: {len(self.data)}")
        print(f"- Features used: {len(self.features.columns)}")
        
        print(f"\nModel Performance:")
        for model_name, results in self.models.items():
            print(f"- {model_name.upper()}:")
            print(f"  * RMSE: {results['rmse']:.4f}")
            print(f"  * MAE: {results['mae']:.4f}")
            print(f"  * R²: {results['r2']:.4f}")
        
        # Best model
        best_model = min(self.models.keys(), key=lambda k: self.models[k]['rmse'])
        print(f"\nBest Model: {best_model.upper()}")
        print(f"RMSE: {self.models[best_model]['rmse']:.4f}")
        
        # Generate forecast
        forecast = self.generate_forecast(best_model)
        
        return best_model, forecast


class StockPriceClassifier:
    """
    A comprehensive stock price classification system with MLflow integration.
    
    This class predicts whether the stock price will increase by at least 5%
    on the next trading day (binary classification).
    """
    
    def __init__(self, stock_symbol, experiment_name="stock_classification"):
        """
        Initialize the classifier.
        
        Args:
            stock_symbol (str): Stock ticker symbol (e.g., 'BBCA.JK')
            experiment_name (str): MLflow experiment name
        """
        self.stock_symbol = stock_symbol
        self.experiment_name = experiment_name
        self.data = None
        self.features = None
        self.target = None
        self.scaler = StandardScaler()
        self.models = {}
        
        # Initialize MLflow
        mlflow.set_experiment(self.experiment_name)
        
    def load_data(self, period="2y", interval="1d"):
        """
        Load stock data using yfinance.
        
        Args:
            period (str): Data period (e.g., '2y', '5y')
            interval (str): Data interval (e.g., '1d', '1h')
        """
        try:
            print(f"Loading data for {self.stock_symbol}...")
            ticker = yf.Ticker(self.stock_symbol)
            self.data = ticker.history(period=period, interval=interval)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.stock_symbol}")
                
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
            
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def engineer_features(self):
        """
        Create technical indicators and lag features for classification.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        print("\n=== Feature Engineering for Classification ===")
        df = self.data.copy()
        
        # Basic price features
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
        df['Price_Change'] = df['Close'].pct_change()
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}']
        
        # Volatility features
        df['Volatility_5'] = df['Close'].rolling(window=5).std()
        df['Volatility_20'] = df['Close'].rolling(window=20).std()
        
        # Technical indicators (same as regression model)
        if TALIB_AVAILABLE:
            try:
                df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
                macd, macd_signal, macd_hist = talib.MACD(df['Close'].values)
                df['MACD'] = macd
                df['MACD_Signal'] = macd_signal
                df['MACD_Hist'] = macd_hist
                bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'].values)
                df['BB_Upper'] = bb_upper
                df['BB_Lower'] = bb_lower
                df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
                df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
                df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'].values, 
                                                          df['Low'].values, 
                                                          df['Close'].values)
            except Exception as e:
                print(f"Error with TA-Lib indicators: {e}")
        else:
            # Manual technical indicators
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            bb_middle = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = bb_middle + (bb_std * 2)
            df['BB_Lower'] = bb_middle - (bb_std * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / bb_middle
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Return_Lag_{lag}'] = df['Price_Change'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'Close_Rolling_Max_{window}'] = df['Close'].rolling(window=window).max()
            df[f'Close_Rolling_Min_{window}'] = df['Close'].rolling(window=window).min()
            df[f'Volume_Rolling_Mean_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # Classification target: 1 if next day's price ≥ 5% higher, 0 otherwise
        df['Next_Day_Close'] = df['Close'].shift(-1)
        df['Price_Increase_5pct'] = ((df['Next_Day_Close'] - df['Close']) / df['Close'] >= 0.015).astype(int)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['Price_Increase_5pct', 'Next_Day_Close', 'Open', 'High', 'Low', 'Close', 'Volume']]
        self.features = df[feature_cols]
        self.target = df['Price_Increase_5pct']
        
        print(f"Features created: {len(feature_cols)}")
        print(f"Final dataset shape: {df.shape}")
        print(f"Target distribution: {self.target.value_counts().to_dict()}")
        print(f"Positive class ratio: {self.target.mean():.3f}")
        
        return self.features, self.target
    
    def create_baseline_model(self, y_test):
        """
        Create simple baseline models for comparison.
        
        Args:
            y_test: Test target values
        """
        print("\n=== Creating Baseline Models ===")
        
        # Random baseline (based on class distribution in training)
        positive_ratio = self.target.mean()
        random_pred = np.random.choice([0, 1], size=len(y_test), p=[1-positive_ratio, positive_ratio])
        random_acc = accuracy_score(y_test, random_pred)
        
        # Majority class baseline (always predict most common class)
        majority_class = self.target.mode()[0]
        majority_pred = np.full(len(y_test), majority_class)
        majority_acc = accuracy_score(y_test, majority_pred)
        
        baselines = {
            'random': {'accuracy': random_acc},
            'majority_class': {'accuracy': majority_acc}
        }
        
        print(f"Random baseline accuracy: {random_acc:.4f}")
        print(f"Majority class baseline accuracy: {majority_acc:.4f}")
        
        return baselines
    
    def train_models(self, test_size=0.2, random_state=42):
        """
        Train multiple classification models and track experiments with MLflow.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        if self.features is None or self.target is None:
            raise ValueError("Features not engineered. Call engineer_features() first.")
        
        print("\n=== Training Classification Models ===")
        
        # Split data (time series aware)
        split_idx = int(len(self.features) * (1 - test_size))
        X_train = self.features.iloc[:split_idx]
        X_test = self.features.iloc[split_idx:]
        y_train = self.target.iloc[:split_idx]
        y_test = self.target.iloc[split_idx:]
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create baseline models
        baselines = self.create_baseline_model(y_test)
        
        # Define models to train
        models_config = {
            'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        }
        
        # Add XGBoost if available
        if XGB_AVAILABLE:
            models_config['xgboost'] = xgb.XGBClassifier(n_estimators=100, random_state=random_state)
        
        # Add LightGBM if available
        if LGB_AVAILABLE:
            models_config['lightgbm'] = lgb.LGBMClassifier(n_estimators=100, random_state=random_state, verbose=-1)
        
        best_model = None
        best_f1 = 0
        
        # Train each model
        for model_name, model in models_config.items():
            with mlflow.start_run(run_name=f"{model_name}_{self.stock_symbol}_classifier"):
                print(f"\nTraining {model_name}...")
                
                # Log parameters
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("stock_symbol", self.stock_symbol)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("train_size", len(X_train))
                mlflow.log_param("positive_class_ratio", y_train.mean())
                
                # Use scaled data for logistic regression, original for tree-based
                if model_name in ['logistic_regression']:
                    X_train_model = X_train_scaled
                    X_test_model = X_test_scaled
                else:
                    X_train_model = X_train
                    X_test_model = X_test
                
                # Train model
                model.fit(X_train_model, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_model)
                y_pred_proba = model.predict_proba(X_test_model)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                try:
                    auc_roc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc_roc = 0.5  # If ROC AUC can't be calculated
                
                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("auc_roc", auc_roc)
                
                # Log baseline comparison
                mlflow.log_metric("baseline_random_acc", baselines['random']['accuracy'])
                mlflow.log_metric("baseline_majority_acc", baselines['majority_class']['accuracy'])
                mlflow.log_metric("improvement_over_majority", accuracy - baselines['majority_class']['accuracy'])
                
                # Log model
                if model_name == 'xgboost' and XGB_AVAILABLE:
                    mlflow.xgboost.log_model(model,name= "model")
                elif model_name == 'lightgbm' and LGB_AVAILABLE:
                    mlflow.lightgbm.log_model(model,name= "model")
                else:
                    mlflow.sklearn.log_model(model,name= "model")
                
                # Create and log confusion matrix plot
                os.makedirs('classifier_image', exist_ok=True)
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'{model_name} - Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plot_path = f'classifier_image/{model_name}_confusion_matrix.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(plot_path)
                plt.close()
                
                # Store model and results
                self.models[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_roc': auc_roc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                # Track best model based on F1 score
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model_name
                
                print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        print(f"\nBest model: {best_model} (F1: {best_f1:.4f})")
        
        # Register best model
        with mlflow.start_run(run_name=f"best_classifier_{self.stock_symbol}"):
            mlflow.log_param("best_model", best_model)
            mlflow.log_metric("best_f1_score", best_f1)
            
            best_model_obj = self.models[best_model]['model']
            if best_model == 'xgboost' and XGB_AVAILABLE:
                mlflow.xgboost.log_model(best_model_obj,name= "best_model", registered_model_name=f"{self.stock_symbol}_classifier")
            elif best_model == 'lightgbm' and LGB_AVAILABLE:
                mlflow.lightgbm.log_model(best_model_obj,name= "best_model", registered_model_name=f"{self.stock_symbol}_classifier")
            else:
                mlflow.sklearn.log_model(best_model_obj,name= "best_model", registered_model_name=f"{self.stock_symbol}_classifier")
        
        return self.models, best_model
    
    def generate_prediction(self, model_name=None):
        """
        Generate prediction for the next trading day.
        
        Args:
            model_name (str): Name of the model to use for prediction
        """
        if not self.models:
            raise ValueError("No models trained. Call train_models() first.")
        
        if model_name is None:
            # Use the best model (highest F1 score)
            model_name = max(self.models.keys(), key=lambda k: self.models[k]['f1_score'])
        
        model = self.models[model_name]['model']
        
        # Get the latest features
        latest_features = self.features.iloc[-1:].copy()
        
        # Scale if needed
        if model_name in ['logistic_regression']:
            latest_features_scaled = self.scaler.transform(latest_features)
            prediction = model.predict(latest_features_scaled)[0]
            prob = model.predict_proba(latest_features_scaled)[0][1]
        else:
            prediction = model.predict(latest_features)[0]
            prob = model.predict_proba(latest_features)[0][1]
        
        current_price = self.data['Close'].iloc[-1]
        
        print(f"\n=== Classification Prediction Results ===")
        print(f"Model used: {model_name}")
        print(f"Current price: {current_price:.2f}")
        print(f"Prediction: {'📈 Price will increase ≥5%' if prediction == 1 else '📉 Price will NOT increase ≥5%'}")
        print(f"Confidence (probability): {prob:.3f}")
        
        return prediction, prob
    
    def create_summary_report(self):
        """Create a comprehensive summary report."""
        if not self.models:
            raise ValueError("No models trained. Call train_models() first.")
        
        print("\n" + "="*60)
        print(f"STOCK PRICE CLASSIFICATION REPORT - {self.stock_symbol}")
        print("="*60)
        
        print(f"\nData Summary:")
        print(f"- Period: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        print(f"- Total samples: {len(self.data)}")
        print(f"- Features used: {len(self.features.columns)}")
        print(f"- Positive class ratio: {self.target.mean():.3f}")
        
        print(f"\nModel Performance:")
        for model_name, results in self.models.items():
            print(f"- {model_name.upper()}:")
            print(f"  * Accuracy: {results['accuracy']:.4f}")
            print(f"  * Precision: {results['precision']:.4f}")
            print(f"  * Recall: {results['recall']:.4f}")
            print(f"  * F1-Score: {results['f1_score']:.4f}")
            print(f"  * AUC-ROC: {results['auc_roc']:.4f}")
        
        # Best model
        best_model = max(self.models.keys(), key=lambda k: self.models[k]['f1_score'])
        print(f"\nBest Model: {best_model.upper()}")
        print(f"F1-Score: {self.models[best_model]['f1_score']:.4f}")
        
        # Generate prediction
        prediction, prob = self.generate_prediction(best_model)
        
        return best_model, prediction, prob


def main():
    """
    Main function to run both stock price regression and classification pipelines.
    """
    # Configuration
    STOCK_SYMBOL = "BBCA.JK"  # Bank Central Asia (Indonesian stock)
    
    try:
        print("🏦 INDONESIAN STOCK ANALYSIS PIPELINE")
        print("="*60)
        
        # =====================
        # REGRESSION PIPELINE
        # =====================
        print("\n🎯 STARTING REGRESSION PIPELINE (Price Forecasting)")
        print("-" * 50)
        
        # Initialize regressor
        regressor = StockPriceRegressor(STOCK_SYMBOL, "indonesian_stock_regression")
        
        # Load and explore data
        regressor.load_data(period="2y")
        regressor.explore_data()
        
        # Engineer features
        regressor.engineer_features()
        
        # Train models
        regression_models, best_regression_model = regressor.train_models()
        
        # Generate summary report
        regressor.create_summary_report()
        
        # =====================
        # CLASSIFICATION PIPELINE
        # =====================
        print("\n\n🎯 STARTING CLASSIFICATION PIPELINE (5% Increase Prediction)")
        print("-" * 50)
        
        # Initialize classifier
        classifier = StockPriceClassifier(STOCK_SYMBOL, "indonesian_stock_classification")
        
        # Load data (reuse same data loading logic)
        classifier.load_data(period="2y")
        
        # Engineer features for classification
        classifier.engineer_features()
        
        # Train classification models
        classification_models, best_classification_model = classifier.train_models()
        
        # Generate summary report
        classifier.create_summary_report()
        
        # =====================
        # FINAL SUMMARY
        # =====================
        print("\n\n" + "="*80)
        print("🎉 COMPLETE ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n📈 REGRESSION RESULTS:")
        print(f"   Best Model: {best_regression_model.upper()}")
        print(f"   RMSE: {regression_models[best_regression_model]['rmse']:.4f}")
        print(f"   Next Day Price Forecast: Available ✅")
        
        print(f"\n📊 CLASSIFICATION RESULTS:")
        print(f"   Best Model: {best_classification_model.upper()}")
        print(f"   F1-Score: {classification_models[best_classification_model]['f1_score']:.4f}")
        print(f"   5% Increase Prediction: Available ✅")
        
        print(f"\n🔬 MLFLOW EXPERIMENTS:")
        print(f"   Regression Experiment: 'indonesian_stock_regression'")
        print(f"   Classification Experiment: 'indonesian_stock_classification'")
        
       
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        raise


def run_regression():
    """
    Function to run the regression pipeline.
    """
    STOCK_SYMBOL = "BBCA.JK"
    
    try:
        print("🎯 RUNNING REGRESSION PIPELINE")
        print("="*50)
        
        regressor = StockPriceRegressor(STOCK_SYMBOL, "stock_regression_forecasting")
        regressor.load_data(period="2y",interval='1d')
        regressor.explore_data()
        regressor.engineer_features()
        regressor.train_models()
        regressor.create_summary_report()
        
        
        print("\n" + "="*80)
        print("✅ Regression pipeline completed!")
        print("="*80)
        print("🌐 Check MLflow UI for detailed experiment tracking:")
        print("   Run: mlflow server --host 127.0.0.1 --port 8080")
        print("   Then visit: http://localhost:8080")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error in regression pipeline: {e}")
        raise

if __name__ == "__main__":
    # You can run different pipelines by changing the function call:
    
    # Run both regression and classification (default)
    # main()
    
    # Or run only regression:
    run_regression_only()
    
    # Or run only classification:
    # run_classification_only()