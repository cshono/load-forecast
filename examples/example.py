from forecasting_library.forecasting.forecasting_api import ForecastingAPI
from forecasting_library.forecasting.preprocessing import preprocess_data, create_preprocessing_pipeline
from forecasting_library.datasets import load_caiso_data

# Step 1: Load data
data = load_caiso_data()

# Step 2: Split Data into Training and Test Sets
# For demonstration, use a simple train-test split; backtesting will be shown later
train_data = data.iloc[:-24]
test_data = data.iloc[-24:]

# Step 3: Separate Features and Target
X_train, y_train = preprocess_data(train_data, target_column='CAISO_system_load')
X_test, y_test = preprocess_data(train_data, target_column='CAISO_system_load')

# Step 4: Create the Preprocessing Pipeline12
preprocessing_pipeline = create_preprocessing_pipeline()

# Step 6: Preprocess the Data
# Fit the preprocessing pipeline on the training data and transform both training and test sets
X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)
X_test_preprocessed = preprocessing_pipeline.transform(X_test)

# Initialize the forecasting API with a linear model
# forecasting_api = ForecastingAPI(model_type='linear')
#forecasting_api = ForecastingAPI(model_type='random_forest')
categorical_features = ["hour"]
forecasting_api = ForecastingAPI(
    model_type = 'grid_search',
    categorical_features = categorical_features,
    numeric_features = [c for c in X_train if c not in categorical_features]
)

# Train the model
forecasting_api.train_model(X_train, y_train)

# Forecast on X_test
predictions = forecasting_api.forecast(X_test)

# Evaluate the model on X_test, y_test
evaluation_metrics = forecasting_api.evaluate_model(X_test, y_test)
print(evaluation_metrics)

# Perform backtesting
X, y = preprocess_data(data, 'CAISO_system_load')
backtest_results = forecasting_api.perform_backtesting(X, y, preprocessing_pipeline)
print(backtest_results)
