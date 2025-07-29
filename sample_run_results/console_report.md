```bash
🎯 RUNNING REGRESSION PIPELINE
==================================================
2025/07/29 18:31:55 INFO mlflow.tracking.fluent: Experiment with name '[2025-07-29] stock_regression_forecasting' does not exist. Creating a new experiment.
Loading data for BBCA.JK...
Data loaded successfully. Shape: (1200, 7)
Date range: 2020-07-29 to 2025-07-29

=== Data Exploration ===
Data shape: (1200, 7)
Missing values:
Open            0
High            0
Low             0
Close           0
Volume          0
Dividends       0
Stock Splits    0
dtype: int64

Data types:
Open            float64
High            float64
Low             float64
Close           float64
Volume            int64
Dividends       float64
Stock Splits    float64
dtype: object

Basic statistics:
               Open          High           Low         Close        Volume    Dividends  Stock Splits
count   1200.000000   1200.000000   1200.000000   1200.000000  1.200000e+03  1200.000000   1200.000000
mean    7715.036679   7782.369203   7646.521941   7713.462568  8.837748e+07     0.855000      0.004167
std     1450.905496   1460.669066   1446.464010   1452.523646  5.935710e+07    11.931287      0.144338
min     4806.284981   4899.224194   4770.879567   4797.433105  1.065390e+07     0.000000      0.000000
25%     6571.382128   6639.128476   6511.595372   6571.382324  5.702390e+07     0.000000      0.000000
50%     7941.094616   8010.148217   7872.042878   7971.554932  7.626740e+07     0.000000      0.000000
75%     8799.237718   8876.866162   8707.878660   8798.982422  1.017658e+08     0.000000      0.000000
max    10522.147296  10570.414456  10401.480891  10570.414062  7.564316e+08   250.000000      5.000000

=== Feature Engineering ===
TA-Lib indicators added successfully
Features created: 42
Final dataset shape: (1150, 48)
Feature columns: ['Dividends', 'Stock Splits', 'High_Low_Pct', 'Price_Change', 'MA_5', 'MA_5_ratio', 'MA_10', 'MA_10_ratio', 'MA_20', 'MA_20_ratio']...

=== Training Models ===
Training set size: 1035
Test set size: 115

=== Creating Baseline Model(s) ===
Moving average RMSE: 205.7659

Training linear_regression...
linear_regression - RMSE: 164.6530, MAE: 125.8647, R²: 0.8146

Training random_forest...
random_forest - RMSE: 213.5445, MAE: 168.3006, R²: 0.6881

Training gradient_boosting...
gradient_boosting - RMSE: 207.0558, MAE: 161.6780, R²: 0.7068

Training xgboost...
xgboost - RMSE: 227.3087, MAE: 180.3070, R²: 0.6466

Training lightgbm...
lightgbm - RMSE: 213.5368, MAE: 166.5430, R²: 0.6881

Best model: linear_regression (RMSE: 164.6530)
Successfully registered model 'BBCA.JK_regressor'.
Created version '1' of model 'BBCA.JK_regressor'.

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

================================================================================
✅ Regression pipeline completed!
================================================================================
🌐 Check MLflow UI for detailed experiment tracking:
   Run: mlflow server --host 127.0.0.1 --port 8080
   Then visit: http://localhost:8080
================================================================================
```