from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline that includes imputation, scaling,
    and feature engineering.
    
    Returns:
        Pipeline: A scikit-learn Pipeline object for preprocessing.
    """
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),   # Impute missing values
        ('scaler', StandardScaler())                   # Scale features
    ])
    return pipeline

def add_time_features(df):
    """Add time-based features to the dataframe."""
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    return df

def preprocess_data(data, target_column):
    """
    Preprocess the data by adding time features, imputing missing values,
    and separating features from the target.
    
    Args:
        data (pd.DataFrame): The input data including features and target.
        target_column (str): The name of the target column in the data.
        imputation_strategy (str): The strategy for imputing missing values.
    
    Returns:
        pd.DataFrame, pd.Series: The preprocessed features and target.
    """
    # Add time-based features
    data = add_time_features(data)

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X, y
