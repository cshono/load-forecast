Extensibility in Future: 
- Used for diff data
- Used for diff evaluation criteria 
- Used for different timing 

Task output can compare:
- models (DONE) 
- hyperparameters (DONE) 
- features (DONE) 

API Expose methods:
- training models (DONE)
- generating forecasts (DONE)
- backtesting (DONE)
- etc. (DONE, plotting) 

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

