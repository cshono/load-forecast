from .base_model import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

class GridSearchModel(BaseModel):
    def __init__(self, categorical_features, numeric_features, n_splits):
        categorical_transformer = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
                #, ("selector", SelectPercentile(chi2, percentile=50))
            ]
        )
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median"))
                , ("std_scaler", StandardScaler())
                , ("minmax_scaler", MinMaxScaler())
                #, ('selector', SelectKBest(mutual_info_regression, k=4))
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        training_pipe = Pipeline(
            steps=[("preprocessor", preprocessor), ('regressor', LinearRegression())]
        )
        
        search_space = [
            {
                #'selector__k': [3,4],# 12, 25, 50],
                'regressor': [LinearRegression()]
                },
            {  
                #'selector__k': [3,4], #, 12, 25, 50],
                #'selector__n_features_to_select': [5,10],
                'regressor': [GradientBoostingRegressor(loss='squared_error')], 
                'regressor__learning_rate': [0.1],
                'regressor__n_estimators': [50, 100]
                }
        ]
        
        timesplit_cv = TimeSeriesSplit(n_splits=n_splits)
        self.model = GridSearchCV(training_pipe, search_space, cv=timesplit_cv, verbose=0) 

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
