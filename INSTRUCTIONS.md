# Take home challenge

The goal of this challenge is to evaluate two things: your python software development skills and your ability to apply 
machine learning to a forecasting problem. To evaluate these skills, we ask you to develop a forecasting library and 
apply it to a specific task.

# Forecasting library

Please create a simple python library that can be used to develop and evaluate forecasts. Imagine that this 
library, if more fully matured, would be used for a variety of forecasting tasks (different data, timings, evaluation 
criteria, etc). For any given task, it would enable the comparison of a variety of models, hyperparameters, or features. 
The library should therefore expose appropriate APIs for functionality that you consider necessary for the task (like 
methods for training models, generating forecasts, backtesting, etc.). Please prioritize code design, readability and
extensibility over the feature completeness necessary to achieve high accuracy (although we do expect you to choose reasonable
model(s) and features for the forecasting task at hand).

Our intent is that you spend only about 4 hours on this challenge. A forecasting library could be arbitrarily complex.
Please descope complexities as you see fit - for example you could assume a univariate target variable. Please update
the `README.md` documenting any assumptions you make and enlightening us of your thought process (e.g. for model selection)
as you see fit. Feel free to describe any additional features you would add to the library if you had more time.

# Forecasting task

Imagine you are a member of the CAISO forecasting team and that you specifically have the responsibility to forecast hourly 
system load (the total hourly electricity demand across  California, in MW or MWh) for the next day prior to the day-ahead
market closing (10 AM). This means developing a model that will run at (or slightly before) 10 am each day D and forecast
from midnight to midnight for day D+1. You can assume historical system load values are available up to the most recent
interval at any point in time. 

Please find included an incomplete (or an initial commit of a) forecasting library with just the functionality to load a small
dataset to help you forecast the CAISO system load. We've also included 6 exogenous weather variables for you to use as you
see fit. The included weather variables are forecasts of temperature and dewpoint at 3 different locations in California (hence
a total of 6 variables) with the same timings:
- These forecasts were generated at 10 am on day D for the midnight-midnight hours of D+1 and stitched together to form individual
    time series (per weather field & location)

Please include a notebook in the `examples` directory to walk a user through the usage of your forecasting library. This
notebook should demonstrate the following:
- As you deem appropriate, include any basic exploratory data analysis that may be informative for your decision(s) related
    to feature engineering, model selection, etc.
- Develop two models that achieve the forecast task above and compare their performance to each other
- Forecast the final 24 hours of the dataset
- How do you expect your model to perform, in general?
    - Include metrics and / or visualizations to support your answer
