from .linear_model import LinearModel
from .random_forest_model import RandomForestModel
from .gridsearch_model import GridSearchModel
from .backtesting import backtest

class ForecastingAPI:
    def __init__(self, model_type='linear', *args, **kwargs):
        """
        Initialize the forecasting API with a specified model type.
        
        Args:
            model_type (str): Type of model to use ('linear' or 'random_forest').
        """
        if model_type == 'linear':
            self.model = LinearModel()
        elif model_type == 'random_forest':
            self.model = RandomForestModel()
        elif model_type == 'grid_search':
            categorical_features = kwargs['categorical_features']
            numeric_features = kwargs['numeric_features']
            self.model = GridSearchModel(categorical_features, numeric_features, n_splits=5)
        else:
            raise ValueError("Unsupported model type. Choose 'linear', 'random_forest', or 'grid_search.")
    
    def train_model(self, X_train, y_train):
        """Train the model using the provided training data."""
        self.model.train(X_train, y_train)
    
    def forecast(self, X):
        """Generate forecasts using the trained model."""
        return self.model.forecast(X)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model's performance."""
        return self.model.evaluate(X_test, y_test)
    
    def perform_backtesting(self, X, y, preprocessing_pipeline, n_splits=5):
        """Perform backtesting to evaluate the model's robustness."""
        return backtest(self.model, X, y, preprocessing_pipeline, n_splits)
