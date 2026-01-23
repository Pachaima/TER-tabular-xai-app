import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from utils import data_manager, model_engine, xai_engine, llm_service

# Page Config
st.set_page_config(page_title="Tabular XAI with Conflict Resolution", layout="wide")

st.title("🔍 Explainable AI on Tabular Data")
st.markdown("### Upload Data, Train Models, and Resolve XAI Conflicts")

# Initialize Session State
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "feature_names" not in st.session_state:
    st.session_state.feature_names = None

# --- SIDEBAR: Data & Model ---
with st.sidebar:
    st.header("1. Data Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = data_manager.load_data(uploaded_file)
            st.session_state.df = df
            st.success("Data Loaded!")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            
    if st.session_state.df is not None:
        st.header("2. Configuration")
        target_col = st.selectbox("Select Target Variable", st.session_state.df.columns)
        
        problem_type = st.radio("Problem Type", ["Classification", "Regression"])
        model_type = st.selectbox("Select Model", ["Random Forest", "Logistic Regression", "Gradient Boosting"] if problem_type == "Classification" else ["Random Forest", "Linear Regression", "Gradient Boosting"])
        
        if st.button("Train Model"):
            with st.spinner("Training..."):
                try:
                    X_train, X_test, y_train, y_test, features = data_manager.preprocess_data(st.session_state.df, target_col)
                    model = model_engine.train_model(X_train, y_train, model_type, problem_type)
                    
                    st.session_state.model = model
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.feature_names = features
                    st.session_state.problem_type = problem_type
                    st.success("Training Complete!")
                except Exception as e:
                    st.error(f"Training Failed: {e}")

# --- MAIN PAGE ---
if st.session_state.model is not None:
    st.divider()
    st.subheader("Model Performance")
    
    # Simple Accuracy/Score
    score = st.session_state.model.score(st.session_state.X_test, st.session_state.y_test)
    st.metric(label="Test Score (Accuracy/R2)", value=f"{score:.4f}")
    
    st.divider()
    st.subheader("3. Explain & Resolve")
    
    # Instance Selection
    row_idx = st.number_input("Select Instance Index to Explain", min_value=0, max_value=len(st.session_state.X_test)-1, value=0)
    
    if st.button("Generate Experience"):
        instance = st.session_state.X_test.iloc[[row_idx]]
        
        with st.spinner("Computing XAI (SHAP & LIME)... This may take a moment."):
            try:
                # 1. Compute SHAP
                shap_vals, expected_val = xai_engine.compute_shap(
                    st.session_state.model, 
                    st.session_state.X_train, 
                    instance,
                    st.session_state.problem_type
                )
                
                # 2. Compute LIME
                lime_exp = xai_engine.compute_lime(
                    st.session_state.model, 
                    st.session_state.X_train, 
                    instance, 
                    st.session_state.feature_names,
                    st.session_state.problem_type
                )
                
                # 3. Consistency
                consistency_score, comparison_df = xai_engine.calculate_consistency(
                    shap_vals, lime_exp, st.session_state.feature_names
                )
                
                # 4. LLM Narrative
                narrative = llm_service.synthesize_explanation(shap_vals, lime_exp, consistency_score, comparison_df)
                
                # --- VISUALIZATION ---
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### SHAP Explanation")
                    # SHAP Plot
                    fig1, ax1 = plt.subplots()
                    # Force plot is complex in Streamlit static, let's use summary/bar for single instance or waterfall
                    # Waterfall is best for single instance
                    try:
                        # Need Explainer object usually for waterfall, providing simple bar instead
                        shap.bar_plot(shap_vals[0] if isinstance(shap_vals, list) else shap_vals, feature_names=st.session_state.feature_names, show=False)
                        st.pyplot(plt.gcf())
                    except Exception as e:
                        st.warning(f"Could not render SHAP plot: {e}")

                
                with col2:
                    st.markdown("#### LIME Explanation")
                    # LIME Plot
                    try:
                        fig2 = lime_exp.as_pyplot_figure()
                        st.pyplot(fig2)
                    except:
                        st.write("LIME Plot unavailable")
                
                st.divider()
                st.markdown("### 🤖 Synthesis & Conflict Resolution")
                
                # Comparison Table
                with st.expander("View Feature Importance Comparison"):
                    st.dataframe(comparison_df)
                
                # Narrative Box
                st.info(narrative)
                
            except Exception as e:
                st.error(f"Error generating explanation: {e}")
                import traceback
                st.code(traceback.format_exc())

else:
    st.info("👈 Please upload a dataset and train a model to begin.")
