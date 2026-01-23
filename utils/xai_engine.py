import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def compute_shap(model, X_train, X_instance, task_type="Classification"):
    """
    Computes SHAP values.
    Uses KernelExplainer as a generic fallback.
    """
    # Summary background data (kmeans to reduce compute)
    # Using small background for speed
    background = shap.kmeans(X_train, 10) if len(X_train) > 100 else X_train
    
    explainer = shap.KernelExplainer(model.predict_proba if task_type == "Classification" else model.predict, background)
    shap_values = explainer.shap_values(X_instance)
    
    # For classification, shap_values is a list [class_0, class_1]. We usually want class_1.
    if task_type == "Classification":
        # Check if binary (2 classes)
        if isinstance(shap_values, list):
            # Return values for the positive class (usually index 1)
            # If binary, index 1 is positive.
            if len(shap_values) == 2:
                return shap_values[1], explainer.expected_value[1]
            else:
                 # Multiclass, just return class 0 for now or handle better
                return shap_values[0], explainer.expected_value[0]
        return shap_values, explainer.expected_value
    else:
        return shap_values, explainer.expected_value

def compute_lime(model, X_train, X_instance, feature_names, task_type="Classification", class_names=None):
    """
    Computes LIME explanation.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=class_names,
        mode="classification" if task_type == "Classification" else "regression"
    )
    
    # LIME expects a 1D array
    instance_array = np.array(X_instance).flatten()
    
    exp = explainer.explain_instance(
        data_row=instance_array,
        predict_fn=model.predict_proba if task_type == "Classification" else model.predict
    )
    return exp

def calculate_consistency(shap_vals, lime_exp, feature_names):
    """
    Calculates consistency between SHAP and LIME.
    Returns: Consistency Score (-1 to 1), Rank DataFrame
    """
    # Process SHAP to get feature importance mapping
    # shap_vals is usually np.array. If it's a list (classification), we handled that above.
    
    # Handle SHAP
    if isinstance(shap_vals, np.ndarray):
         # Flatten if single instance
         shap_vals = shap_vals.flatten()
    
    shap_dict = dict(zip(feature_names, shap_vals))
    
    # Process LIME
    # LIME exp.as_list() returns [(feature_name_condition, weight), ...]
    # We need to map back to original feature names.
    # LIME modifies names (e.g. "age > 20"). We need a robust way or just use the local importance map if available.
    # exp.local_exp[1] gives list of (index, weight). Index matches X_train columns.
    
    # Getting LIME local importance by index
    # local_exp is a dict {label: [(index, weight), ...]}
    # We use label 1 for binary classification usually.
    # If regression, key is likely dummy.
    
    lime_weights_map = {}
    
    # Attempt to retrieve local_exp
    try:
        # Get the first available key (class)
        available_keys = list(lime_exp.local_exp.keys())
        # Ideally we want the positive class.
        target_key = available_keys[-1] # Usually 1 in binary [0, 1]
        
        lime_list = lime_exp.local_exp[target_key]
        for idx, weight in lime_list:
            fname = feature_names[idx]
            lime_weights_map[fname] = weight
            
    except Exception as e:
        print(f"Error parsing LIME: {e}")
        return 0.0, pd.DataFrame()

    # Create comparison dataframe
    features = feature_names
    shap_ranks = []
    lime_ranks = []
    
    data = []
    
    for f in features:
        s_val = shap_dict.get(f, 0)
        l_val = lime_weights_map.get(f, 0)
        data.append({
            "Feature": f,
            "SHAP_Val": s_val,
            "LIME_Val": l_val,
            "SHAP_Abs": abs(s_val),
            "LIME_Abs": abs(l_val)
        })
        
    df_comp = pd.DataFrame(data)
    
    # Rank by Absolute Importance (Descending)
    df_comp["SHAP_Rank"] = df_comp["SHAP_Abs"].rank(ascending=False)
    df_comp["LIME_Rank"] = df_comp["LIME_Abs"].rank(ascending=False)
    
    # Calculate Correlation
    if len(df_comp) > 1:
        corr, _ = spearmanr(df_comp["SHAP_Rank"], df_comp["LIME_Rank"])
    else:
        corr = 1.0
        
    return corr, df_comp
