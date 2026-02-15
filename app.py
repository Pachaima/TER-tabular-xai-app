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

from utils import data_manager, model_engine, xai_engine, llm_service

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

st.title("🏆 TALENT-Integrated XAI Platform")

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
tab_eda, tab_train, tab_xai = st.tabs(["📊 EDA", "🏋️ Benchmarking", "🧠 Explainability"])

# --- TAB 1: EDA ---
with tab_eda:
    if st.session_state.df is not None:
        st.markdown("#### Exploratory Data Analysis")
        
        # Descriptive Stats
        with st.expander("Descriptive Statistics", expanded=True):
            st.dataframe(st.session_state.df.describe())
            st.write(f"**Missing Values:**")
            st.dataframe(st.session_state.df.isnull().sum().to_frame("Count").T)
        
        # Visualizations
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("**Distributions (Numeric)**")
            num_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                selected_num = st.selectbox("Select Feature", num_cols, key="eda_hist")
                # Altair Histogram
                chart = alt.Chart(st.session_state.df).mark_bar().encode(
                    alt.X(selected_num, bin=True),
                    y='count()'
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
        
        with col_viz2:
            st.markdown("**Correlation Heatmap**")
            if num_cols:
                corr = st.session_state.df[num_cols].corr()
                fig, ax = plt.subplots(figsize=(5,4))
                sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
    else:
         st.info("Upload data to view EDA.")

# --- TAB 2: Benchmarking ---
with tab_train:
    st.markdown("#### Model Benchmarking")
    
    if st.session_state.info:
        col_glass, col_black = st.columns(2)
        
        # Glass Box Selections
        with col_glass:
            st.markdown('<div class="card"><h5>🔍 Glass Box Models</h5><p>Interpretable, simple models.</p></div>', unsafe_allow_html=True)
            glass_options = ['Logistic Regression', 'Decision Tree']
            if st.session_state.info['task_type'] == 'regression':
                glass_options = ['Linear Regression', 'Decision Tree']
            selected_glass = st.multiselect("Select Glass Box", glass_options, default=glass_options[:1])
            
        # Black Box Selections
        with col_black:
            st.markdown('<div class="card"><h5>⚫ Black Box Models (TALENT)</h5><p>State-of-the-art Deep Learning models.</p></div>', unsafe_allow_html=True)
            black_options = ['FT-Transformer', 'ResNet', 'SwitchTab', 'TabNet', 'TabR', 'TabPFN']
            selected_black = st.multiselect("Select Black Box", black_options, default=['FT-Transformer', 'ResNet'])
        
        all_selected = selected_glass + selected_black
        
        if st.button("🚀 Start Training", type="primary"):
            results = []
            progress = st.progress(0)
            
            for i, model_name in enumerate(all_selected):
                try:
                    # Train
                    model = model_engine.train_model(
                        st.session_state.X_train, st.session_state.y_train, 
                        st.session_state.X_val, st.session_state.y_val, 
                        model_name, st.session_state.info['task_type']
                    )
                    st.session_state.trained_models[model_name] = model
                    
                    # Evaluate
                    metrics = model_engine.evaluate_model(
                        model, st.session_state.X_val, st.session_state.y_val, st.session_state.info['task_type']
                    )
                    metrics['Model'] = model_name
                    metrics['Type'] = "Glass" if model_name in glass_options else "Black Box"
                    results.append(metrics)
                    
                except Exception as e:
                    st.error(f"Error {model_name}: {e}")
                
                progress.progress((i+1)/len(all_selected))
            
            if results:
                st.session_state.leaderboard = pd.DataFrame(results).set_index("Model")
                st.success("Training Complete!")
        
        # Leaderboard Display
        if not st.session_state.leaderboard.empty:
            st.markdown("### 🏅 Comparison Result")
            
            # Dynamic metric highlighting
            desired_metrics = ['Accuracy', 'F1 Score', 'ROC AUC', 'R2', 'RMSE']
            valid_metrics = [m for m in desired_metrics if m in st.session_state.leaderboard.columns]
            
            st.dataframe(
                st.session_state.leaderboard.style.highlight_max(axis=0, subset=valid_metrics, color='lightgreen'), 
                use_container_width=True
            )
            
            # Champion Selection
            st.divider()
            st.markdown("#### Select Champion")
            champ = st.selectbox("Choose Best Model", st.session_state.leaderboard.index)
            if st.button("Set as Champion"):
                st.session_state.champion_model = st.session_state.trained_models[champ]
                st.success(f"Champion set to: {champ}")

    else:
        st.info("Please process data first.")

# --- TAB 3: XAI ---
with tab_xai:
    if st.session_state.champion_model:
        st.markdown(f"#### Explaining: {type(st.session_state.champion_model).__name__}")
        
        # Get Instance
        if st.session_state.X_test is not None:
            instance = st.session_state.X_test.iloc[[row_idx]]
            
            st.markdown("**Selected Instance Data:**")
            st.dataframe(instance)
            
            if st.button("🔍 Analyze Prediction"):
                with st.spinner("Generating explanations..."):
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
                        # Consistency
                        consistency, df_comp = xai_engine.calculate_consistency(
                            shap_vals, lime_exp, st.session_state.info['feature_names']
                        )
                        
                        # Visuals
                        c1, c2 = st.columns(2)
                        
                        with c1:
                            st.markdown("##### SHAP (Global/Feature Contribution)")
                            # Use bar plot for cleaner look in app
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sv_to_plot = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
                            if isinstance(sv_to_plot, np.ndarray) and sv_to_plot.ndim > 1:
                                 sv_to_plot = sv_to_plot[0] 
                            shap.bar_plot(sv_to_plot, feature_names=st.session_state.info['feature_names'], show=False)
                            st.pyplot(plt.gcf())
                            
                        with c2:
                            st.markdown("##### LIME (Local Linear)")
                            st.pyplot(lime_exp.as_pyplot_figure())
                            
                        # Detail
                        st.divider()
                        st.markdown("##### 📝 Feature Impact Detail")
                        st.dataframe(df_comp.style.background_gradient(cmap='coolwarm', subset=['SHAP_Val', 'LIME_Val']))
                        
                    except Exception as e:
                        st.error(f"XAI Error: {e}")
                        st.write(e)
        else:
             st.warning("No test data found.")
    else:
        st.info("Please select a champion model in the Benchmarking tab first.")
