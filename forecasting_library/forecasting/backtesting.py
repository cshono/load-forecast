from sklearn.model_selection import TimeSeriesSplit

def backtest(model, X, y, preprocessing_pipeline, n_splits=5):
    """
    Perform backtesting with a given model and preprocessing pipeline.
    
    Args:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        model: The forecasting model to be used (e.g., a LinearModel instance).
        preprocessing_pipeline: The preprocessing pipeline to apply.
        n_splits (int): The number of splits for time series cross-validation.
    
    Returns:
        list: A list of mean absolute errors for each fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for train_index, test_index in tscv.split(X):
        # Split the data into training and testing sets for this fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit the preprocessing pipeline on the training data
        X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)
        X_test_preprocessed = preprocessing_pipeline.transform(X_test)

        # Train the model
        model.train(X_train_preprocessed, y_train)
        
        # Evaluate the model
        evaluation_metrics = model.evaluate(X_test_preprocessed, y_test)
        results.append(evaluation_metrics)
    
    return results
