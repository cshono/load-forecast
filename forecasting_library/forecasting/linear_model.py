from .base_model import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import numpy as np

class LinearModel(BaseModel):
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        """Train the Linear Regression model."""
        self.model.fit(X_train, y_train)

    def forecast(self, X):
        """Generate forecasts using the trained Linear Regression model."""
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate the Linear Regression model performance."""
        y_pred = self.forecast(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse}
