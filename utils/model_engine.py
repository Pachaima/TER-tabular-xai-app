from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train, model_type="Random Forest", problem_type="Classification"):
    """
    Trains a model based on the selected type.
    """
    model = None
    
    if problem_type == "Classification":
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(random_state=42)
            
    elif problem_type == "Regression":
         if model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
         elif model_type == "Linear Regression":
            model = LinearRegression()
         elif model_type == "Gradient Boosting":
            model = GradientBoostingRegressor(random_state=42)
            
    if model:
        model.fit(X_train, y_train)
        
    return model

def predict(model, X):
    """Returns predictions."""
    return model.predict(X)

def predict_proba(model, X):
    """Returns prediction probabilities (for classification)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None
