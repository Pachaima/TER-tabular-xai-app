import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

def show_eda_page(df=None):
    """
    Function to display the Pre-modelling Explainability page.
    It takes a pandas DataFrame as input.
    """
    st.header("Pre-modelling Explainability")
    
    if df is None:
        st.info("Please upload a dataset in the sidebar to view analysis.")
        return

    # 1. Dataset Overview
    st.subheader("1. Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    
    # Missing Values
    missing_data = df.isnull().sum()
    total_missing = missing_data.sum()
    col3.metric("Missing Values", total_missing)

    with st.expander("Show Raw Data"):
        st.dataframe(df.head())
    
    with st.expander("Data Types"):
        st.write(df.dtypes)
        
    with st.expander("Descriptive Statistics"):
        st.write(df.describe(include='all'))

    # 2. Target Variable Analysis
    st.subheader("2. Target Variable Analysis")
    target_col = st.selectbox("Select Target Variable", df.columns, key="eda_target")
    
    if target_col:
        st.markdown(f"**Distribution of {target_col}**")
        fig, ax = plt.subplots(figsize=(8, 4))
        if pd.api.types.is_numeric_dtype(df[target_col]):
            sns.histplot(df[target_col], kde=True, ax=ax)
            st.pyplot(fig)
            task_type = 'regression'
        else:
            sns.countplot(x=df[target_col], ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            task_type = 'classification'

        # 3. Features Visualization
        st.subheader("3. Feature Visualization")
        
        # Identify numerical and categorical columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        viz_type = st.radio("Choose visualization type", ["Univariate", "Bivariate", "Correlation Matrix", "Mutual Information"], horizontal=True)

        if viz_type == "Univariate":
            selected_col = st.selectbox("Select Column", df.columns)
            if selected_col:
                 fig, ax = plt.subplots(figsize=(8, 4))
                 if selected_col in numeric_cols:
                     sns.histplot(df[selected_col], kde=True, ax=ax)
                     ax.set_title(f"Distribution of {selected_col}")
                 else:
                     sns.countplot(y=df[selected_col], ax=ax, order=df[selected_col].value_counts().index)
                     ax.set_title(f"Count of {selected_col}")
                 st.pyplot(fig)

        elif viz_type == "Bivariate":
            selected_col = st.selectbox("Select Feature to compare with Target", [c for c in df.columns if c != target_col])
            if selected_col:
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # Numeric Feature vs Numeric Target (Scatter)
                if selected_col in numeric_cols and target_col in numeric_cols:
                    sns.scatterplot(data=df, x=selected_col, y=target_col, ax=ax)
                    ax.set_title(f"{selected_col} vs {target_col}")
                
                # Numeric Feature vs Categorical Target (Boxplot)
                elif selected_col in numeric_cols and target_col in categorical_cols:
                    sns.boxplot(data=df, x=target_col, y=selected_col, ax=ax)
                    ax.set_title(f"{selected_col} by {target_col}")

                # Categorical Feature vs Numeric Target (Boxplot)
                elif selected_col in categorical_cols and target_col in numeric_cols:
                    sns.boxplot(data=df, x=selected_col, y=target_col, ax=ax)
                    plt.xticks(rotation=45)
                    ax.set_title(f"{target_col} by {selected_col}")
                
                # Categorical vs Categorical (Countplot with hue)
                else:
                    sns.countplot(data=df, x=selected_col, hue=target_col, ax=ax)
                    plt.xticks(rotation=45)
                    ax.set_title(f"{selected_col} distribution by {target_col}")
                
                st.pyplot(fig)

        elif viz_type == "Correlation Matrix":
            if len(numeric_cols) > 1:
                st.write("Correlation Heatmap (Numerical columns only)")
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = df[numeric_cols].corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Not enough numerical columns for correlation matrix.")
                
        elif viz_type == "Mutual Information":
             st.markdown("**Mutual Information Scores (Non-linear Dependency Estimate)**")
             with st.spinner("Calculating Mutual Information..."):
                 # Prepare data for MI
                 df_mi = df.copy()
                 # Encode categoricals
                 for col in categorical_cols:
                     le = LabelEncoder()
                     df_mi[col] = le.fit_transform(df_mi[col].astype(str))
                 
                 # Drop rows with NaNs in target for calculation
                 df_mi = df_mi.dropna(subset=[target_col])
                 
                 X_mi = df_mi.drop(columns=[target_col])
                 y_mi = df_mi[target_col]
                 
                 # Encode target if classification
                 if task_type == 'classification':
                    le_target = LabelEncoder()
                    y_mi = le_target.fit_transform(y_mi)
                    mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42)
                 else:
                    mi_scores = mutual_info_regression(X_mi, y_mi, random_state=42)
                 
                 mi_df = pd.DataFrame({'Feature': X_mi.columns, 'Mutual Information': mi_scores})
                 mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)
                 
                 fig, ax = plt.subplots(figsize=(8, len(mi_df)*0.4 + 2))
                 sns.barplot(data=mi_df, y='Feature', x='Mutual Information', ax=ax, palette='viridis')
                 ax.set_title("Feature Importance by Mutual Information")
                 st.pyplot(fig)
    else:
        st.info("Select a Target Variable to enable deeper analysis.")
