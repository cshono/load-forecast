from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


class TimeFeaturesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_column="index"):
        """
        Custom transformer to add time-based features.

        Args:
            datetime_column (str): The name of the datetime column.
        """
        self.datetime_column = datetime_column

    def fit(self, X, y=None):
        # No fitting required for adding time-based features
        return self

    def transform(self, X):
        X = X.copy()
        if self.datetime_column == "index":
            X[self.datetime_column] = X.index
        X["hour"] = X[self.datetime_column].dt.hour
        X["day_of_week"] = X[self.datetime_column].dt.dayofweek
        X["month"] = X[self.datetime_column].dt.month
        return X.drop(columns=[self.datetime_column])


def create_preprocessing_pipeline(numeric_cols, categorical_cols, datetime_column="index"):
    """
    Create a preprocessing pipeline that includes imputation, scaling,
    and feature engineering.

    Returns:
        Pipeline: A scikit-learn Pipeline object for preprocessing.
    """
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
            # , ("selector", SelectPercentile(chi2, percentile=50))
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler()),
            ("minmax_scaler", MinMaxScaler()),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return Pipeline(
        steps=[
            ("add_time_features", TimeFeaturesAdder(datetime_column=datetime_column)),
            ("column_transformer", column_transformer),
        ]
    )
