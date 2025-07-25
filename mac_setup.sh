#!/bin/bash

echo "🍎 Mac Setup Script for Stock Price Forecasting"
echo "=============================================="

# Check if Homebrew is installed
# if ! command -v brew &> /dev/null; then
#     echo "❌ Homebrew not found. Installing Homebrew..."
#     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# else
#     echo "✅ Homebrew found"
# fi

# Update Homebrew
echo "📦 Updating Homebrew..."
brew update

# # Install Python if not available
# if ! command -v python3 &> /dev/null; then
#     echo "🐍 Installing Python..."
#     brew install python@3.9
# else
#     echo "✅ Python found"
# fi

# Install core requirements
echo "📋 Installing core requirements..."
pip3 install yfinance pandas numpy scikit-learn mlflow matplotlib seaborn requests

# Try to install XGBoost
echo "🚀 Attempting to install XGBoost..."
if pip3 install xgboost; then
    echo "✅ XGBoost installed successfully"
else
    echo "❌ XGBoost installation failed. The script will work without it."
fi

# Try to install LightGBM
echo "💡 Attempting to install LightGBM..."
if pip3 install lightgbm; then
    echo "✅ LightGBM installed successfully"
else
    echo "❌ LightGBM installation failed. The script will work without it."
fi

# Try to install TA-Lib
echo "📊 Attempting to install TA-Lib..."
if brew install ta-lib && pip3 install TA-Lib; then
    echo "✅ TA-Lib installed successfully"
else
    echo "❌ TA-Lib installation failed. Using manual technical indicators instead."
fi

echo ""
echo "🎉 Setup complete!"
echo "You can now run: python3 stock_forecasting.py"
echo ""
echo "If some packages failed to install, don't worry - the script will work with fallbacks."