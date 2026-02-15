import pandas as pd
import numpy as np
import os
import json
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(file_buffer):
    """Loads a CSV or Parquet file into a Pandas DataFrame."""
    try:
        if file_buffer.name.endswith('.parquet'):
            return pd.read_parquet(file_buffer)
        else:
            return pd.read_csv(file_buffer)
    except Exception as e:
        return None

def preprocess_data(df, target_column, output_dir="data/current_dataset", random_state=42):
    """
    Preprocesses data for TALENT and Classical models.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test (Pandas Objects)
        feature_info (dict)
    """
    # 1. Clean Data
    df = df.dropna(subset=[target_column])
    
    # 2. Identify Features
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Detect Task Type
    if y.dtype == 'object' or len(y.unique()) < 20: 
        # Heuristic: if object or few unique values -> Classification
        if len(y.unique()) == 2:
            task_type = "binclass"
        else:
            task_type = "multiclass"
        
        # Encode Target
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y), name=target_column)
        target_mapping = {int(k): str(v) for k, v in enumerate(le_target.classes_)}
    else:
        task_type = "regression"
        # For regression, we still usually keep y as is, but TALENT usually expects float
        y = y.astype(float)
        target_mapping = None

    # Detect Feature Types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Fill NA (Simple Imputation)
    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].mean())
    for col in categorical_features:
        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Missing")
    
    # Encode Categorical Features consistently
    cat_mappings = {}
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        cat_mappings[col] = {int(k): str(v) for k, v in enumerate(le.classes_)}
    
    # 3. Split Data (Train/Val/Test -> 70/15/15)
    stratify = y if task_type != "regression" else None
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=stratify
    )
    
    stratify_temp = y_temp if task_type != "regression" else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=stratify_temp
    )
    
    # Helper to extract N/C parts for saving
    def get_parts(df_split):
        n_part = df_split[numeric_features].to_numpy().astype(np.float32)
        c_part = df_split[categorical_features].to_numpy().astype(np.int32)
        return n_part, c_part

    N_train, C_train = get_parts(X_train)
    N_val, C_val = get_parts(X_val)
    N_test, C_test = get_parts(X_test)
    
    # 4. Serialize to .npy
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, 'N_train.npy'), N_train)
    np.save(os.path.join(output_dir, 'N_val.npy'), N_val)
    np.save(os.path.join(output_dir, 'N_test.npy'), N_test)
    
    np.save(os.path.join(output_dir, 'C_train.npy'), C_train)
    np.save(os.path.join(output_dir, 'C_val.npy'), C_val)
    np.save(os.path.join(output_dir, 'C_test.npy'), C_test)
    
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train.to_numpy())
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val.to_numpy())
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test.to_numpy())
    
    # 5. Generate info.json
    info = {
        "task_type": task_type,
        "n_num_features": len(numeric_features),
        "n_cat_features": len(categorical_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_names": X.columns.tolist(),
        "cat_mappings": cat_mappings,
        "target_mapping": target_mapping
    }
    
    with open(os.path.join(output_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)
        
    return X_train, X_val, X_test, y_train, y_val, y_test, info
