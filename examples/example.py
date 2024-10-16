# Example usage in the notebook
from forecasting_library.datasets import load_caiso_data 
from forecasting_library.forecasting.preprocessing import create_preprocessing_pipeline
from forecasting_library.forecasting.forecasting_api import ForecastingAPI
from forecasting_library.forecasting.evaluation import backtest_model

data = load_caiso_data()
X = data.drop(columns=['CAISO_system_load'])
y = data['CAISO_system_load'] 

# Step 1: Create the Preprocessing Pipeline
categorical_cols = ["hour"] 
numeric_cols = [c for c in X if c not in categorical_cols] 
preprocessing_pipeline = create_preprocessing_pipeline(numeric_cols, categorical_cols, datetime_column='index')

# Step 1: Choose a More Sophisticated Model
model_type = 'xgboost'

# Step 2: Define the Hyperparameter Grid for the Model
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

# Step 3: Initialize the ForecastingAPI with the Preprocessing Pipeline
forecasting_api = ForecastingAPI(model_type=model_type, preprocessing_pipeline=preprocessing_pipeline)

# Step 4: Split training and test data. Split off the final month (jul 2023) for test 
X_train, X_test, y_train, y_test = forecasting_api.split_train_test(data, target_col="CAISO_system_load", split_date="2022-08-01")

# Step 4: Train the Model Using the Complete Training Pipeline (from pre jul 2023 data)
mae = forecasting_api.train_pipeline(X_train, y_train, param_grid=param_grid, test_size=0.2)

# Step 5: Make Predictions on New Data 
predictions = forecasting_api.forecast(X_test)

# Step 6: Run Backtesting
import numpy as np  
backtest_results = backtest_model(forecasting_api.model, X, y, n_splits=14, test_size=24)
average_mape = np.mean([r['MAPE'] for r in backtest_results])
print(f'Average Mean Absolute Percent Error (MAPE) during backtesting: {average_mape}')


'''


# Step 2: Define the Model and Parameter Grid
model_type = 'random_forest'
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}

# Step 3: Initialize the ForecastingAPI with the Preprocessing Pipeline
forecasting_api = ForecastingAPI(model_type=model_type, preprocessing_pipeline=preprocessing_pipeline)

# Step 4: Split training and test data. Split off the final month (jul 2023) for test 
X_train, X_test, y_train, y_test = forecasting_api.split_train_test(data, target_col="CAISO_system_load", split_date="2022-08-01")

# Step 4: Train the Model Using the Complete Training Pipeline (from pre jul 2023 data)
mae = forecasting_api.train_pipeline(X_train, y_train, param_grid=param_grid, test_size=0.2)

# Step 5: Make Predictions on New Data 
predictions = forecasting_api.forecast(X_test)

# Step 6: Run Backtesting
import numpy as np  
backtest_results = backtest_model(forecasting_api.model, X, y, n_splits=14, test_size=24)
average_mape = np.mean([r['MAPE'] for r in backtest_results])
print(f'Average Mean Absolute Percent Error (MAPE) during backtesting: {average_mape}')
'''



