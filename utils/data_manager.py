import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_buffer):
    """Loads a CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_buffer)
        return df
    except Exception as e:
        return None

def preprocess_data(df, target_column, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    Handles basic preprocessing: drops rows with missing target.
    Returns: X_train, X_test, y_train, y_test, feature_names, class_names (if classification)
    """
    # Drop rows where target is missing
    df = df.dropna(subset=[target_column])
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify feature types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Simple encoding for categorical features (One-Hot Encoding)
    # in a real app, we would use a pipeline to handle this more robustly
    # For now, we get dummies to ensure models can handle it
    X_processed = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    # Impute missing values with mean for numeric (simplified)
    # Using pandas fillna for simplicity in this demo script
    X_processed = X_processed.fillna(X_processed.mean())
    
    feature_names = X_processed.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, feature_names
