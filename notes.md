Extensibility in Future: 
- Used for diff data
- Used for diff evaluation criteria 
- Used for different timing 

Task output can compare:
- models 
- hyperparameters 
- features 

API Expose methods:
- training models
- generating forecasts 
- backtesting 
- etc. 

Put assumptions in README.md:
- Univariate target variable 
- etc. 

Forecasting Task:
- Forecast hourly system load (MWH or MW) 
- Next day prior to day ahead market closing (10 AM) 
- Each day D, forecast midnight to midnight for day D+1 
- Assume historical load values are available up to most recent interval 

Examples:
- include an example notebook to walk a user through the usage of the forecasting library
- include any EDA related to feature engineering, model selection, etc. 
- Develop 2 models that achieve the forecast task above and compare their performance to each other 
- How do you expect your model to perform, in general? 
    - Include metrics and/or visualizations to support your answer 


columns: 
feature1
feature2
feature1_lag_1
feature1_lag_2
feature2_lag_1
feature2_lag_2
hour_1
hour_2
hour_3
hour_4 
target
target_lag_1
target_lag_2
