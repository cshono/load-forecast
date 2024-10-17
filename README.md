# My Forecasting Library
This library contains a core forecast model for generating day-ahead or other short-term load forecasts. 

## Forecasting Modules:
### forecast_model.py
ForecastModel: Domain model for representing a load forecast 

### forecasting_task.py
ForecastingTask: API for executing forecast tasks/workflows using the ForecastModel 

### preprocessing.py
Module for preprocessing and feature engineering the raw data. 

### evaluation.py 
Module for evaluating forecast predictions. 

## Assumption:
- univariate target
- assumes Regression-based (non-classifier models) BaseEstimator for the numeric target.
- assumes feature values are numeric (non-categorical). This holds true for the provided weather and derived time features. Future additions to the preprocessing module could include onehot-encoding for categorical features 
- assumes sufficient data quality of feature coverage to dropna samples. Would need to implement an imputer class if training data coverage becomes too sparse. 
- target lags are not allowed to be less than 38 in the training features because the day-ahead forecast must predict a predictions for all 24 hours of D+1 at 10 AM of day D. There is 38 hour difference between 10 AM on day D and midnight on day D+1. 
- the ForecastingTask wrapper class does not prohibit an input of target_lag < 38 because the library could potentially be extended to shorter horizon use cases such as a real-time electricity market forecasts. 


# Contributing

## Getting Started

Ensure you have `poetry` installed. You can do this via `pip install poetry` if needed.

```
make init
```

To use the `poetry` virtual environment, you can either run all your code inside a `poetry shell` or run commands using `poetry run ...`.

## Checking your code

```
# Formatting / Linting
make format

# Type checking
make typecheck

# Unit tests
make run-tests
```
