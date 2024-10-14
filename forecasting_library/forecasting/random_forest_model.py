from .base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def train(self, X_train, y_train):
        """Train the Random Forest Regressor model."""
        self.model.fit(X_train, y_train)

    def forecast(self, X):
        """Generate forecasts using the trained Random Forest model."""
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Evaluate the Random Forest model performance."""
        y_pred = self.forecast(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return {'MAE': mae, 'RMSE': rmse}
