from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model with the provided training data."""
        pass

    @abstractmethod
    def forecast(self, X):
        """Generate forecasts using the trained model."""
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """Evaluate the model on the test set."""
        pass
