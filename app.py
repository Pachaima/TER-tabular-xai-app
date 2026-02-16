import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import data_manager, model_engine, xai_engine, llm_service, eda_engine

# --- Page Config & CSS ---
st.set_page_config(page_title="TALENT XAI Platform", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for "Cards" and Clean Design
st.markdown("""
<style>
    .card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #eee;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("TALENT-Integrated XAI Platform")

# Initialize Session State
if "df" not in st.session_state: st.session_state.df = None
if "X_train" not in st.session_state: st.session_state.X_train = None
if "X_val" not in st.session_state: st.session_state.X_val = None
if "X_test" not in st.session_state: st.session_state.X_test = None
if "y_train" not in st.session_state: st.session_state.y_train = None
if "y_val" not in st.session_state: st.session_state.y_val = None
if "y_test" not in st.session_state: st.session_state.y_test = None
if "info" not in st.session_state: st.session_state.info = None
if "trained_models" not in st.session_state: st.session_state.trained_models = {}
if "leaderboard" not in st.session_state: st.session_state.leaderboard = pd.DataFrame()
if "champion_model" not in st.session_state: st.session_state.champion_model = None

# ==========================================
# SIDEBAR (Control & Data)
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/brain.png", width=100)
    st.header("⚙️ Control Panel")
    
    # Section 1: Data Upload
    st.subheader("1. Data Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = data_manager.load_data(uploaded_file)
            st.session_state.df = df
            st.success(f"Data Loaded: {df.shape}")
        except Exception as e:
            st.error(f"Error: {e}")
            
    # Section 2: Model Selection
    if st.session_state.df is not None:
        st.divider()
        st.subheader("2. Settings")
        target_col = st.selectbox("Target Variable", st.session_state.df.columns)
        
        # Simple wrapper to get task type if not processed
        if st.button("Process Data", use_container_width=True):
            with st.spinner("Processing..."):
                 try:
                     X_tr, X_val, X_te, y_tr, y_val, y_te, info = data_manager.preprocess_data(st.session_state.df, target_col)
                     st.session_state.X_train = X_tr
                     st.session_state.X_val = X_val
                     st.session_state.X_test = X_te
                     st.session_state.y_train = y_tr
                     st.session_state.y_val = y_val
                     st.session_state.y_test = y_te
                     st.session_state.info = info
                     st.success("Done!")
                 except Exception as e:
                     st.error(str(e))
        
        if st.session_state.info:
            st.info(f"Task: {st.session_state.info['task_type']}")

    # Section 3: Instance Selector (Only if X_test exists)
    if st.session_state.X_test is not None:
        st.divider()
        st.subheader("3. Explain Instance")
        row_idx = st.number_input(
            "Instance ID", 
            min_value=0, 
            max_value=len(st.session_state.X_test)-1, 
            value=0
        )


# ==========================================
# MAIN AREA (Workflow)
# ==========================================

# TABS for workflow
tab_pre, tab_intrinsic, tab_post = st.tabs(["📊 Pre-modelling Explainability", "🔍 Intrinsic Interpretation (Glass Box)", "🧠 Post-hoc Explainability (Black Box)"])

# --- TAB 1: Pre-modelling ---
with tab_pre:
    eda_engine.show_eda_page(st.session_state.df)

# --- TAB 2: Intrinsic Interpretation ---
with tab_intrinsic:
    st.markdown("#### Intrinsic Interpretation (Glass Box Models)")
    
    if st.session_state.info:
        col_glass_sel, col_glass_res = st.columns([1, 2])
        
        with col_glass_sel:
            st.markdown('<div class="card"><h5>Models</h5></div>', unsafe_allow_html=True)
            glass_options = ['Logistic Regression', 'Decision Tree', 'KNN']
            if st.session_state.info['task_type'] == 'regression':
                glass_options = ['Linear Regression', 'Decision Tree', 'KNN']
            selected_glass = st.multiselect("Select Models", glass_options, default=glass_options[:2])
            
            if st.button("🚀 Train Glass Box Models", type="primary"):
                results = []
                with st.spinner("Training..."):
                    for model_name in selected_glass:
                        try:
                            # Train
                            model = model_engine.train_model(
                                st.session_state.X_train, st.session_state.y_train, 
                                st.session_state.X_val, st.session_state.y_val, 
                                model_name, st.session_state.info['task_type']
                            )
                            # Save to a specific dict for Glass Box to avoid overwrite/confusion
                            if "glass_models" not in st.session_state: st.session_state.glass_models = {}
                            st.session_state.glass_models[model_name] = model
                            
                            # Evaluate
                            metrics = model_engine.evaluate_model(
                                model, st.session_state.X_val, st.session_state.y_val, st.session_state.info['task_type']
                            )
                            metrics['Model'] = model_name
                            results.append(metrics)
                        except Exception as e:
                            st.error(f"Error {model_name}: {e}")
                
                if results:
                    st.session_state.glass_leaderboard = pd.DataFrame(results).set_index("Model")
                    st.success("Training Complete!")

        with col_glass_res:
             if "glass_leaderboard" in st.session_state and not st.session_state.glass_leaderboard.empty:
                st.markdown("##### Performance Metrics")
                st.dataframe(st.session_state.glass_leaderboard.style.highlight_max(axis=0, color='lightgreen'))
                
                st.divider()
                st.markdown("##### Interpretation")
                # Select model to interpret
                model_to_interpret = st.selectbox("Select Model to Interpret", st.session_state.glass_leaderboard.index)
                chosen_model = st.session_state.glass_models[model_to_interpret]
                
                if "Regression" in model_to_interpret:
                    st.write("**Log-Odds / Coefficients**")
                    df_coeffs = model_engine.get_lr_coeffs(chosen_model, st.session_state.info['feature_names'])
                    if df_coeffs is not None:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.barplot(data=df_coeffs.head(10), x='Coefficient', y='Feature', ax=ax, palette='viridis')
                        st.pyplot(fig)
                    else:
                        st.info("Coefficients not available for this model type.")
                        
                elif "Decision Tree" in model_to_interpret:
                    st.write("**Tree Structure (Text)**")
                    text_rep = model_engine.get_dt_text(chosen_model, st.session_state.info['feature_names'])
                    st.text(text_rep)
                    
                elif "KNN" in model_to_interpret:
                    st.write("**Nearest Neighbors**")
                    if st.session_state.X_test is not None:
                        instance = st.session_state.X_test.iloc[[row_idx]] # Use shared row_idx
                        dist, ind = model_engine.get_knn_neighbors(chosen_model, instance, st.session_state.X_train)
                        
                        st.write(f"Neighbors for Instance {row_idx}:")
                        # Show neighbors from X_train (need to map back to df if possible, or just show values)
                        # We have X_train as dataframe? No, X_train is numpy in data_manager if split? 
                        # Wait, data_manager returns pandas df for X_train.
                        
                        neighbors_df = st.session_state.X_train.iloc[ind].copy()
                        neighbors_df['Distance'] = dist
                        st.dataframe(neighbors_df)
                    else:
                        st.warning("No test data available for instance selection.")

    else:
        st.info("Please upload and process data first.")

# --- TAB 3: Post-hoc Explainability ---
with tab_post:
    st.markdown("#### Post-hoc Explainability (TALENT / Black Box)")
    
    if st.session_state.info:
        col_black_sel, col_black_res = st.columns([1, 2]) # Reuse layout concept
        
        with col_black_sel:
            st.markdown('<div class="card"><h5>TALENT Models</h5></div>', unsafe_allow_html=True)
            black_options = ['FT-Transformer', 'ResNet', 'SwitchTab', 'TabNet', 'TabR', 'TabPFN']
            selected_black = st.multiselect("Select Deep Models", black_options, default=['FT-Transformer', 'ResNet'])
            
            if st.button("🚀 Train & Benchmark", type="primary"):
                results = []
                progress = st.progress(0)
                
                for i, model_name in enumerate(selected_black):
                    try:
                        model = model_engine.train_model(
                            st.session_state.X_train, st.session_state.y_train, 
                            st.session_state.X_val, st.session_state.y_val, 
                            model_name, st.session_state.info['task_type']
                        )
                        st.session_state.trained_models[model_name] = model # Keep separate or same? Same is fine for champion
                        
                        metrics = model_engine.evaluate_model(
                            model, st.session_state.X_val, st.session_state.y_val, st.session_state.info['task_type']
                        )
                        metrics['Model'] = model_name
                        results.append(metrics)
                    except Exception as e:
                        st.error(f"Error {model_name}: {e}")
                    progress.progress((i+1)/len(selected_black))
                
                if results:
                    st.session_state.leaderboard = pd.DataFrame(results).set_index("Model")
                    st.success("Benchmarking Complete!")
                    
                if not st.session_state.leaderboard.empty:
                    st.markdown("##### Leaderboard")
                    valid_metrics = [m for m in ['Accuracy', 'F1 Score', 'ROC AUC', 'R2', 'RMSE'] if m in st.session_state.leaderboard.columns]
                    st.dataframe(st.session_state.leaderboard.style.highlight_max(axis=0, subset=valid_metrics, color='lightgreen'))
                    
                    st.divider()
                    st.markdown("##### Explainability Suite")
                    champ = st.selectbox("Select Champion Model", st.session_state.leaderboard.index)
                    st.session_state.champion_model = st.session_state.trained_models[champ]
                    
                    # Explainability Tabs
                    tab_global, tab_local = st.tabs(["🌍 Global Explainability", "📍 Local Explainability"])
                    
                    # --- Global: Permutation Importance ---
                    with tab_global:
                        if st.button("📉 Calculate Permutation Importance"):
                            with st.spinner("Permuting features... (this may take a while)"):
                                df_imp = xai_engine.compute_permutation_importance(
                                    st.session_state.champion_model,
                                    st.session_state.X_val,
                                    st.session_state.y_val,
                                    st.session_state.info['task_type']
                                )
                                if not df_imp.empty:
                                    st.markdown("**Global Feature Importance (Permutation)**")
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    sns.barplot(data=df_imp, x='Importance', y='Feature', ax=ax, palette='viridis')
                                    ax.set_title(f"Permutation Importance - {champ}")
                                    st.pyplot(fig)
                                    st.dataframe(df_imp)
                                else:
                                    st.error("Failed to calculate importance.")

                    # --- Local: SHAP & LIME ---
                    with tab_local:
                        # Instance Selection (from Sidebar)
                        if st.session_state.X_test is not None:
                             instance = st.session_state.X_test.iloc[[row_idx]]
                             st.write(f"Explaining Instance ID: **{row_idx}**")
                             st.dataframe(instance)
                             
                             if st.button("🔍 Generate Local Explanations"):
                                with st.spinner("Computing SHAP & LIME..."):
                                    try:
                                        # SHAP
                                        shap_vals, expected_val = xai_engine.compute_shap(
                                            st.session_state.champion_model,
                                            st.session_state.X_train,
                                            instance,
                                            st.session_state.info['task_type']
                                        )
                                        # LIME
                                        lime_exp = xai_engine.compute_lime(
                                            st.session_state.champion_model,
                                            st.session_state.X_train,
                                            instance,
                                            st.session_state.info['feature_names'],
                                            st.session_state.info['task_type']
                                        )
                                        
                                        col_shap, col_lime = st.columns(2)
                                        
                                        with col_shap:
                                            st.markdown("**SHAP (Feature Contribution)**")
                                            # Use bar plot for cleaner look in app
                                            fig, ax = plt.subplots(figsize=(6, 4))
                                            sv_to_plot = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
                                            if isinstance(sv_to_plot, np.ndarray) and sv_to_plot.ndim > 1:
                                                 sv_to_plot = sv_to_plot[0] 
                                            shap.bar_plot(sv_to_plot, feature_names=st.session_state.info['feature_names'], show=False)
                                            st.pyplot(plt.gcf())
                                            
                                        with col_lime:
                                            st.markdown("**LIME (Local Surrogate)**")
                                            st.pyplot(lime_exp.as_pyplot_figure())
                                            
                                        # Force Plot (Optional expander)
                                        with st.expander("View Force Plot"):
                                            try:
                                                 shap.plots.force(expected_val[0] if isinstance(expected_val, (list, np.ndarray)) else expected_val, 
                                                                  sv_to_plot, instance, matplotlib=True, show=False)
                                                 st.pyplot(plt.gcf())
                                            except Exception as e:
                                                st.info(f"Force plot unavailable: {e}")
                                                    
                                    except Exception as e:
                                        st.error(f"XAI Error: {e}")
                        else:
                            st.warning("No Test Data.")
    else:
        st.info("Process Data First.")
