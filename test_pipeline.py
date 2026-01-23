import pandas as pd
import numpy as np
from utils import model_engine, xai_engine, llm_service, data_manager

def test_pipeline():
    print("1. Creating Dummy Data...")
    df = pd.DataFrame({
        'age': np.random.randint(20, 60, 100),
        'income': np.random.randint(30000, 100000, 100),
        'credit_score': np.random.randint(300, 850, 100),
        'default': np.random.randint(0, 2, 100)
    })
    
    print("2. Preprocessing...")
    X_train, X_test, y_train, y_test, features = data_manager.preprocess_data(df, 'default')
    print(f"Features: {features}")
    
    print("3. Training Model...")
    model = model_engine.train_model(X_train, y_train, "Random Forest", "Classification")
    print("Model Trained.")
    
    print("4. Predicting...")
    instance = X_test.iloc[[0]]
    pred = model_engine.predict(model, instance)
    print(f"Prediction: {pred}")
    
    print("5. XAI: SHAP...")
    try:
        shap_vals, expected = xai_engine.compute_shap(model, X_train, instance, "Classification")
        print("SHAP Computed.")
    except Exception as e:
        print(f"SHAP Failed: {e}")
        raise e
        
    print("6. XAI: LIME...")
    try:
        lime_exp = xai_engine.compute_lime(model, X_train, instance, features, "Classification")
        print("LIME Computed.")
    except Exception as e:
        print(f"LIME Failed: {e}")
        raise e
        
    print("7. Consistency Check...")
    score, comp_df = xai_engine.calculate_consistency(shap_vals, lime_exp, features)
    print(f"Consistency Score: {score}")
    print(comp_df)
    
    print("8. LLM Synthesis...")
    narrative = llm_service.synthesize_explanation(shap_vals, lime_exp, score, comp_df)
    print("Narrative Generated:")
    print(narrative[:100] + "...")
    
    print("✅ TEST PASSED")

if __name__ == "__main__":
    test_pipeline()
