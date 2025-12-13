import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, classification_report, 
                             confusion_matrix, mean_absolute_error, roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import io
import base64
from datetime import datetime
import json

# Set page config
st.set_page_config(page_title="Synthetic Data Modeling", layout="wide")

# Initialize session state for dark mode
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Custom CSS with dark mode support
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #64b5f6;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #ffb74d;
            margin-top: 2rem;
        }
        .stApp {
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #ff7f0e;
            margin-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Synthetic Data Modeling & Simulation</h1>', unsafe_allow_html=True)

# Dark mode toggle
# Theme toggle removed per project preference

st.markdown("---")

# Sidebar for configuration
st.sidebar.header("Configuration")
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Regression", "Classification"]
)

# Algorithm selection
if model_type == "Regression":
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        ["Linear Regression", "Random Forest", "Decision Tree", "Support Vector Machine"]
    )
else:
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        ["Logistic Regression", "Random Forest", "Decision Tree", "Support Vector Machine"]
    )

# Dataset parameters
st.sidebar.subheader("Dataset Parameters")
n_samples = st.sidebar.slider("Number of Samples", 100, 10000, 1000, 100)
n_features = st.sidebar.slider("Number of Features", 2, 20, 5, 1)
noise = st.sidebar.slider("Noise Level", 0.0, 50.0, 10.0, 1.0)
random_state = st.sidebar.number_input("Random State", 0, 1000, 42)

if model_type == "Classification":
    n_classes = st.sidebar.slider("Number of Classes", 2, 5, 2, 1)
    n_informative = st.sidebar.slider("Informative Features", 2, n_features, min(3, n_features), 1)
else:
    # Defaults when regression is selected (app still generates classification dataset)
    n_classes = 2
    n_informative = min(3, n_features)

# Custom naming options
st.sidebar.subheader("Naming Options")
use_custom_names = st.sidebar.checkbox("Use Custom Names", value=False)

if use_custom_names:
    dataset_name = st.sidebar.text_input("Dataset Name", value="MyDataset")
    target_name = st.sidebar.text_input("Target Variable Name", value="Target")
    
    st.sidebar.markdown("**Feature Names** (comma-separated)")
    custom_features = st.sidebar.text_area(
        "Enter feature names",
        value=", ".join([f"Feature_{i+1}" for i in range(n_features)]),
        help="Enter feature names separated by commas. Number of names should match number of features."
    )

# Generate Data Button
generate_data = st.sidebar.button("Generate Data", type="primary")

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'experiment_history' not in st.session_state:
    st.session_state.experiment_history = []
if 'experiment_counter' not in st.session_state:
    st.session_state.experiment_counter = 0

# Main content
if generate_data or st.session_state.data_generated:
    st.session_state.data_generated = True
    
    # Step 1: Generate Synthetic Dataset
    st.markdown('<h2 class="sub-header">1️⃣ Generate Synthetic Dataset</h2>', unsafe_allow_html=True)
    
    with st.spinner("Generating synthetic data..."):
        # Always generate a dataset that mimics common weather features (base schema)
        np.random.seed(int(random_state) if random_state is not None else None)
        locations = ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Canberra', 'Hobart']
        locs = np.random.choice(locations, size=n_samples)

        MinTemp = np.round(np.random.normal(10, 5, n_samples), 1)
        MaxTemp = np.round(MinTemp + np.abs(np.random.normal(8, 3, n_samples)), 1)
        Temp9am = np.round(MinTemp + np.random.normal(2, 1, n_samples), 1)
        Temp3pm = np.round(MaxTemp + np.random.normal(-1, 1, n_samples), 1)
        Humidity9am = np.clip(np.round(np.random.normal(75, 12, n_samples), 1), 0, 100)
        Humidity3pm = np.clip(np.round(np.random.normal(60, 15, n_samples), 1), 0, 100)
        WindSpeed9am = np.round(np.abs(np.random.normal(15, 5, n_samples)), 1)
        WindSpeed3pm = np.round(np.abs(np.random.normal(18, 6, n_samples)), 1)

        # Rainfall and flags
        rain_prob = np.clip((Humidity9am / 100) * 0.6 + (Humidity3pm / 100) * 0.3, 0, 1)
        rain_events = np.random.rand(n_samples) < rain_prob
        Rainfall = np.where(rain_events, np.round(np.random.exponential(scale=5, size=n_samples), 1), 0.0)
        RainToday = (Rainfall > 0).astype(int)

        # RainTomorrow probability
        logits = -3 + 0.04 * Humidity3pm + 0.2 * (Rainfall > 0) + 0.01 * (20 - Temp3pm)
        prob_tom = 1 / (1 + np.exp(-logits))
        RainTomorrow = (np.random.rand(n_samples) < prob_tom).astype(int)

        df = pd.DataFrame({
            'Location': locs,
            'MinTemp': MinTemp,
            'MaxTemp': MaxTemp,
            'Temp9am': Temp9am,
            'Temp3pm': Temp3pm,
            'Humidity9am': Humidity9am,
            'Humidity3pm': Humidity3pm,
            'WindSpeed9am': WindSpeed9am,
            'WindSpeed3pm': WindSpeed3pm,
            'Rainfall': Rainfall,
            'RainToday': RainToday,
            'RainTomorrow': RainTomorrow
        })

        # Compute average humidity and keep compact dataset for modeling: predictors + target
        df['Humidity'] = ((df['Humidity9am'] + df['Humidity3pm']) / 2).round(1)
        df = df[['Humidity', 'MinTemp', 'MaxTemp', 'Rainfall', 'RainTomorrow']].copy()

        # Ensure numeric types
        df['RainTomorrow'] = df['RainTomorrow'].astype(int)

        # Use Humidity as the modeling feature instead of Location
        feature_names = ['Humidity', 'MinTemp', 'MaxTemp', 'Rainfall']
        target_col = 'RainTomorrow'
        problem_type = 'Classification'
        dataset_title = 'Synthetic Dataset (weather-like base features)'

        X = df[feature_names].values
        y = df[target_col].values

        # Store in session state
        st.session_state.df = df
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.feature_names = feature_names
        st.session_state.target_col = target_col
        st.session_state.dataset_title = dataset_title
        st.session_state.problem_type = problem_type
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", n_samples)
    with col2:
        st.metric("Features", n_features)
    with col3:
        st.metric("Problem Type", problem_type)
    
    st.success(f"Data generated successfully: **{st.session_state.dataset_title}**")
    
    # Download button for generated data
    col_download1, col_download2 = st.columns([3, 1])
    with col_download2:
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        # Use custom dataset name for file if provided
        file_name = st.session_state.dataset_title.lower().replace(' ', '_') if use_custom_names else f'synthetic_data_{problem_type.lower()}_{n_samples}samples'
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f'{file_name}.csv',
            mime='text/csv',
            help="Download the generated dataset as CSV"
        )
    
    with st.expander("View Dataset Sample"):
        st.dataframe(st.session_state.df.head(20), use_container_width=True)
    
    # Step 2: Exploratory Data Analysis (EDA)
    st.markdown('<h2 class="sub-header">2️⃣ Exploratory Data Analysis (EDA)</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Statistical Summary", "Distributions", "Correlations", "Target Analysis"])
    
    with tab1:
        st.subheader("Statistical Summary")
        st.dataframe(st.session_state.df.describe(), use_container_width=True)
    
    with tab2:
        st.subheader("Feature Distributions")
        selected_features = st.multiselect(
            "Select features to visualize",
            st.session_state.feature_names,
            default=st.session_state.feature_names[:min(3, len(st.session_state.feature_names))]
        )
        
        if selected_features:
            cols = st.columns(min(3, len(selected_features)))
            for idx, feature in enumerate(selected_features):
                with cols[idx % 3]:
                    fig = px.histogram(st.session_state.df, x=feature, nbins=30, 
                                     title=f"{feature} Distribution")
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Correlation Matrix")
        corr_matrix = st.session_state.df.corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       aspect='auto',
                       color_continuous_scale='RdBu_r',
                       title="Feature Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top correlations with target
        target_col = st.session_state.target_col
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
        st.subheader(f"Top Features Correlated with {target_col}")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Positive Correlations:**")
            st.dataframe(target_corr.head(5))
        with col2:
            st.write("**Negative Correlations:**")
            st.dataframe(target_corr.tail(5))

        # Quick predictability report (classification only)
        if st.session_state.problem_type == "Classification":
            st.markdown("**Quick Predictability Report (baseline Logistic Regression, 5-fold CV)**")
            try:
                from sklearn.model_selection import cross_val_score
                from sklearn.linear_model import LogisticRegression

                # Use numeric X,y prepared earlier in session state
                X_cv = st.session_state.X
                y_cv = st.session_state.y

                clf = LogisticRegression(max_iter=500)
                acc = cross_val_score(clf, X_cv, y_cv, cv=5, scoring='accuracy').mean()
                prec = cross_val_score(clf, X_cv, y_cv, cv=5, scoring='precision').mean()
                rec = cross_val_score(clf, X_cv, y_cv, cv=5, scoring='recall').mean()
                f1 = cross_val_score(clf, X_cv, y_cv, cv=5, scoring='f1').mean()

                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("CV Accuracy", f"{acc:.3f}")
                with col_b:
                    st.metric("CV Precision", f"{prec:.3f}")
                with col_c:
                    st.metric("CV Recall", f"{rec:.3f}")
                with col_d:
                    st.metric("CV F1", f"{f1:.3f}")

                # ROC AUC if possible
                try:
                    auc_score = cross_val_score(clf, X_cv, y_cv, cv=5, scoring='roc_auc').mean()
                    st.metric("CV ROC AUC", f"{auc_score:.3f}")
                except Exception:
                    pass
            except Exception as e:
                st.warning(f"Quick predictability report unavailable: {e}")
    
    with tab4:
        st.subheader(f"{st.session_state.target_col} Variable Analysis")
        if problem_type == "Classification":
            fig = px.histogram(st.session_state.df, x=st.session_state.target_col, 
                             title=f"{st.session_state.target_col} Class Distribution",
                             color=st.session_state.target_col)
            st.plotly_chart(fig, use_container_width=True)
            
            # Class distribution
            class_dist = st.session_state.df[st.session_state.target_col].value_counts()
            st.write("**Class Distribution:**")
            st.dataframe(class_dist)
        else:
            fig = px.histogram(st.session_state.df, x=st.session_state.target_col, nbins=50,
                             title=f"{st.session_state.target_col} Variable Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean", f"{st.session_state.y.mean():.2f}")
                st.metric("Std Dev", f"{st.session_state.y.std():.2f}")
            with col2:
                st.metric("Min", f"{st.session_state.y.min():.2f}")
                st.metric("Max", f"{st.session_state.y.max():.2f}")
    
    # Step 2.5: Advanced Feature Engineering & Analysis
    st.markdown('<h2 class="sub-header">2.5️⃣ Advanced Feature Engineering & Analysis</h2>', unsafe_allow_html=True)
    
    fe_tabs = st.tabs(["Feature Transforms", "Feature Selection", "Dimensionality Reduction", "Visualizations"])
    
    with fe_tabs[0]:
        st.subheader("Feature Transformations")
        st.write("Apply mathematical transformations to features")
        
        transform_type = st.selectbox(
            "Select Transformation",
            ["None", "Logarithmic (log1p)", "Exponential", "Square Root", "Square"]
        )
        
        if transform_type != "None":
            transform_features = st.multiselect(
                "Select features to transform",
                st.session_state.feature_names,
                default=[]
            )
            
            if transform_features and st.button("Apply Transformation"):
                df_transformed = st.session_state.df.copy()
                for feature in transform_features:
                    if transform_type == "Logarithmic (log1p)":
                        df_transformed[f"{feature}_log"] = np.log1p(df_transformed[feature] - df_transformed[feature].min() + 1)
                    elif transform_type == "Exponential":
                        df_transformed[f"{feature}_exp"] = np.exp(df_transformed[feature])
                    elif transform_type == "Square Root":
                        df_transformed[f"{feature}_sqrt"] = np.sqrt(df_transformed[feature] - df_transformed[feature].min())
                    elif transform_type == "Square":
                        df_transformed[f"{feature}_sq"] = df_transformed[feature] ** 2
                
                st.session_state.df_transformed = df_transformed
                st.success(f"Applied {transform_type} transformation to {len(transform_features)} features")
                st.dataframe(df_transformed.head(), use_container_width=True)
    
    with fe_tabs[1]:
        st.subheader("Automated Feature Selection")
        
        selection_method = st.radio(
            "Select Method",
            ["SelectKBest", "Recursive Feature Elimination (RFE)"]
        )
        
        k_features = st.slider("Number of features to select", 1, n_features, min(5, n_features))
        
        if st.button("Run Feature Selection"):
            with st.spinner("Selecting features..."):
                if selection_method == "SelectKBest":
                    if problem_type == "Classification":
                        selector = SelectKBest(f_classif, k=k_features)
                    else:
                        selector = SelectKBest(f_regression, k=k_features)
                    selector.fit(st.session_state.X, st.session_state.y)
                    scores = selector.scores_
                    selected_mask = selector.get_support()
                else:  # RFE
                    if problem_type == "Classification":
                        estimator = RandomForestClassifier(n_estimators=50, random_state=random_state)
                    else:
                        estimator = RandomForestRegressor(n_estimators=50, random_state=random_state)
                    selector = RFE(estimator, n_features_to_select=k_features)
                    selector.fit(st.session_state.X, st.session_state.y)
                    selected_mask = selector.support_
                    scores = selector.ranking_
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Feature': st.session_state.feature_names,
                    'Selected': selected_mask,
                    'Score/Rank': scores
                }).sort_values('Score/Rank', ascending=(selection_method == "RFE"))
                
                st.dataframe(results_df, use_container_width=True)
                
                # Visualization
                fig = px.bar(results_df, x='Feature', y='Score/Rank', 
                            color='Selected',
                            title=f"Feature {selection_method} Results")
                st.plotly_chart(fig, use_container_width=True)
                
                st.session_state.selected_features = [f for f, s in zip(st.session_state.feature_names, selected_mask) if s]
    
    with fe_tabs[2]:
        st.subheader("PCA - Dimensionality Reduction")
        
        n_components = st.slider("Number of Components", 2, min(10, n_features), 2)
        
        if st.button("Apply PCA"):
            with st.spinner("Performing PCA..."):
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(st.session_state.X)
                
                # Explained variance
                explained_var = pca.explained_variance_ratio_
                cumulative_var = np.cumsum(explained_var)
                
                st.session_state.X_pca = X_pca
                st.session_state.pca = pca
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Scree plot
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[f'PC{i+1}' for i in range(n_components)],
                        y=explained_var * 100,
                        name='Individual'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[f'PC{i+1}' for i in range(n_components)],
                        y=cumulative_var * 100,
                        name='Cumulative',
                        yaxis='y2'
                    ))
                    fig.update_layout(
                        title='Explained Variance by Component',
                        yaxis=dict(title='Variance Explained (%)'),
                        yaxis2=dict(title='Cumulative (%)', overlaying='y', side='right')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # PCA scatter plot
                    pca_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
                    pca_df['Target'] = st.session_state.y
                    
                    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Target',
                                   title='PCA: First Two Components')
                    st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"Total variance explained by {n_components} components: {cumulative_var[-1]*100:.2f}%")
    
    with fe_tabs[3]:
        st.subheader("Visualizations")

        # 2D Visualization
        st.write("**2D Feature Relationships**")
        if n_features >= 2:
            # prepare feature list and sensible defaults (Humidity vs MaxTemp)
            features_2d = list(st.session_state.feature_names)
            x_default_idx = features_2d.index('Humidity') if 'Humidity' in features_2d else 0
            y_default_idx = features_2d.index('MaxTemp') if 'MaxTemp' in features_2d else min(1, len(features_2d)-1)

            col1, col2 = st.columns(2)
            with col1:
                x_feature_2d = st.selectbox("X-axis (2D)", features_2d, index=x_default_idx, key='x2d')
            with col2:
                y_feature_2d = st.selectbox("Y-axis (2D)", features_2d, index=y_default_idx, key='y2d')

            fig2d = px.scatter(
                st.session_state.df,
                x=x_feature_2d,
                y=y_feature_2d,
                color=st.session_state.target_col if st.session_state.target_col in st.session_state.df.columns else None,
                title=f"2D Scatter: {x_feature_2d} vs {y_feature_2d}",
                opacity=0.8
            )
            st.plotly_chart(fig2d, use_container_width=True)
        else:
            st.warning("Need at least 2 features for 2D visualization")

        st.markdown("---")

        # 3D Visualization
        st.write("**3D Feature Relationships**")
        if n_features >= 3:
            # only use modeling features for 3D axes (exclude target like RainTomorrow); default to MinTemp/MaxTemp/Rainfall
            features_3d = list(st.session_state.feature_names)
            x3d_idx = features_3d.index('MinTemp') if 'MinTemp' in features_3d else 0
            y3d_idx = features_3d.index('MaxTemp') if 'MaxTemp' in features_3d else min(1, len(features_3d)-1)
            z3d_idx = features_3d.index('Rainfall') if 'Rainfall' in features_3d else min(2, len(features_3d)-1)

            col1, col2, col3 = st.columns(3)
            with col1:
                x_feature = st.selectbox("X-axis (3D)", features_3d, index=x3d_idx, key='x3d')
            with col2:
                y_feature = st.selectbox("Y-axis (3D)", features_3d, index=y3d_idx, key='y3d')
            with col3:
                z_feature = st.selectbox("Z-axis (3D)", features_3d, index=z3d_idx, key='z3d')

            fig = go.Figure(data=[go.Scatter3d(
                x=st.session_state.df[x_feature],
                y=st.session_state.df[y_feature],
                z=st.session_state.df[z_feature],
                mode='markers',
                marker=dict(
                    size=4,
                    color=st.session_state.y,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Target")
                )
            )])

            fig.update_layout(
                title='3D Feature Space Visualization',
                scene=dict(
                    xaxis_title=x_feature,
                    yaxis_title=y_feature,
                    zaxis_title=z_feature
                ),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 3 features for 3D visualization")
    
    # Step 3: Apply Modeling Technique
    st.markdown('<h2 class="sub-header">3️⃣ Apply Modeling Technique</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 50, 20, 5) / 100
    with col2:
        scale_features = st.checkbox("Scale Features", value=True)
    
    train_model = st.button("Train Model", type="primary")
    
    if train_model or st.session_state.model_trained:
        st.session_state.model_trained = True
        
        with st.spinner("Training model..."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state.X, st.session_state.y,
                test_size=test_size,
                random_state=random_state
            )
            
            # Scale features
            if scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                st.session_state.scaler = scaler
            
            # Train model
            # Ensure selected algorithm matches problem type; fall back to sensible defaults
            if problem_type == "Classification" and algorithm not in ["Logistic Regression", "Random Forest", "Decision Tree", "Support Vector Machine"]:
                st.warning("Selected algorithm is incompatible with classification target; defaulting to Logistic Regression.")
                algorithm = "Logistic Regression"
            if problem_type == "Regression" and algorithm not in ["Linear Regression", "Random Forest", "Decision Tree", "Support Vector Machine"]:
                st.warning("Selected algorithm is incompatible with regression target; defaulting to Linear Regression.")
                algorithm = "Linear Regression"

            if problem_type == "Regression":
                if algorithm == "Linear Regression":
                    model = LinearRegression()
                elif algorithm == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
                elif algorithm == "Decision Tree":
                    model = DecisionTreeRegressor(random_state=random_state)
                elif algorithm == "Support Vector Machine":
                    model = SVR(kernel='rbf')
                
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                st.session_state.model = model
                st.session_state.algorithm = algorithm
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.y_pred_train = y_pred_train
                st.session_state.y_pred_test = y_pred_test
            else:
                if algorithm == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000, random_state=random_state)
                elif algorithm == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                elif algorithm == "Decision Tree":
                    model = DecisionTreeClassifier(random_state=random_state)
                elif algorithm == "Support Vector Machine":
                    model = SVC(kernel='rbf', random_state=random_state)
                
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                st.session_state.model = model
                st.session_state.algorithm = algorithm
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.y_pred_train = y_pred_train
                st.session_state.y_pred_test = y_pred_test
        
        st.success(f"Model trained successfully using {st.session_state.algorithm}!")
        
        # Step 4: Model Evaluation
        st.markdown('<h2 class="sub-header">4️⃣ Model Evaluation</h2>', unsafe_allow_html=True)
        st.info(f"**Algorithm Used:** {st.session_state.algorithm}")
        
        if problem_type == "Regression":
            # Metrics
            train_mse = mean_squared_error(st.session_state.y_train, st.session_state.y_pred_train)
            test_mse = mean_squared_error(st.session_state.y_test, st.session_state.y_pred_test)
            train_mae = mean_absolute_error(st.session_state.y_train, st.session_state.y_pred_train)
            test_mae = mean_absolute_error(st.session_state.y_test, st.session_state.y_pred_test)
            train_r2 = r2_score(st.session_state.y_train, st.session_state.y_pred_train)
            test_r2 = r2_score(st.session_state.y_test, st.session_state.y_pred_test)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Train MSE", f"{train_mse:.2f}")
                st.metric("Test MSE", f"{test_mse:.2f}")
            with col2:
                st.metric("Train MAE", f"{train_mae:.2f}")
                st.metric("Test MAE", f"{test_mae:.2f}")
            with col3:
                st.metric("Train R²", f"{train_r2:.4f}")
                st.metric("Test R²", f"{test_r2:.4f}")
            
            # Prediction plots
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.y_train,
                    y=st.session_state.y_pred_train,
                    mode='markers',
                    name='Training',
                    marker=dict(color='blue', opacity=0.5)
                ))
                fig.add_trace(go.Scatter(
                    x=[st.session_state.y_train.min(), st.session_state.y_train.max()],
                    y=[st.session_state.y_train.min(), st.session_state.y_train.max()],
                    mode='lines',
                    name='Perfect Fit',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title="Training Set: Actual vs Predicted",
                    xaxis_title="Actual Values",
                    yaxis_title="Predicted Values"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.y_test,
                    y=st.session_state.y_pred_test,
                    mode='markers',
                    name='Testing',
                    marker=dict(color='green', opacity=0.5)
                ))
                fig.add_trace(go.Scatter(
                    x=[st.session_state.y_test.min(), st.session_state.y_test.max()],
                    y=[st.session_state.y_test.min(), st.session_state.y_test.max()],
                    mode='lines',
                    name='Perfect Fit',
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title="Test Set: Actual vs Predicted",
                    xaxis_title="Actual Values",
                    yaxis_title="Predicted Values"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Residuals
            residuals_train = st.session_state.y_train - st.session_state.y_pred_train
            residuals_test = st.session_state.y_test - st.session_state.y_pred_test
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(x=residuals_train, nbins=30, 
                                 title="Training Residuals Distribution")
                fig.update_layout(xaxis_title="Residuals", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(x=residuals_test, nbins=30,
                                 title="Test Residuals Distribution")
                fig.update_layout(xaxis_title="Residuals", yaxis_title="Frequency")
                st.plotly_chart(fig, use_container_width=True)
            
            # Advanced Regression Visualizations
            st.markdown("---")
            st.subheader("Advanced Analysis")
            
            adv_tabs = st.tabs(["Residual Plots", "Learning Curves", "Feature Importance"])
            
            with adv_tabs[0]:
                col1, col2 = st.columns(2)
                with col1:
                    # Residuals vs Fitted
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=st.session_state.y_pred_test,
                        y=residuals_test,
                        mode='markers',
                        marker=dict(color='blue', opacity=0.5)
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    fig.update_layout(
                        title="Residuals vs Fitted Values",
                        xaxis_title="Fitted Values",
                        yaxis_title="Residuals"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Q-Q Plot
                    from scipy import stats
                    residuals_sorted = np.sort(residuals_test)
                    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals_sorted)))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=theoretical_quantiles,
                        y=residuals_sorted,
                        mode='markers',
                        name='Residuals'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                        y=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                        mode='lines',
                        name='Normal',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title="Q-Q Plot",
                        xaxis_title="Theoretical Quantiles",
                        yaxis_title="Sample Quantiles"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with adv_tabs[1]:
                # Learning Curves
                with st.spinner("Generating learning curves..."):
                    train_sizes = np.linspace(0.1, 1.0, 10)
                    train_sizes_abs, train_scores, test_scores = learning_curve(
                        st.session_state.model, st.session_state.X_train, st.session_state.y_train,
                        train_sizes=train_sizes, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
                    )
                    
                    train_scores_mean = -train_scores.mean(axis=1)
                    train_scores_std = train_scores.std(axis=1)
                    test_scores_mean = -test_scores.mean(axis=1)
                    test_scores_std = test_scores.std(axis=1)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=train_sizes_abs, y=train_scores_mean,
                        name='Training Score',
                        mode='lines+markers',
                        line=dict(color='blue'),
                        error_y=dict(type='data', array=train_scores_std, visible=True)
                    ))
                    fig.add_trace(go.Scatter(
                        x=train_sizes_abs, y=test_scores_mean,
                        name='Cross-validation Score',
                        mode='lines+markers',
                        line=dict(color='green'),
                        error_y=dict(type='data', array=test_scores_std, visible=True)
                    ))
                    fig.update_layout(
                        title="Learning Curves",
                        xaxis_title="Training Examples",
                        yaxis_title="Mean Squared Error",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("Learning curves show how model performance changes with training set size. Converging lines suggest the model is well-tuned.")
            
            with adv_tabs[2]:
                # Feature Importance
                if hasattr(st.session_state.model, 'coef_'):
                    coef_df = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Coefficient': st.session_state.model.coef_
                    }).sort_values('Coefficient', key=abs, ascending=False)
                    
                    fig = px.bar(coef_df, x='Feature', y='Coefficient',
                                title="Feature Coefficients (Linear Model)")
                    st.plotly_chart(fig, use_container_width=True)
                elif hasattr(st.session_state.model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Importance': st.session_state.model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='Feature', y='Importance',
                                title="Feature Importance (Tree-based Model)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type")
        
            
        else:  # Classification
            # Metrics
            train_acc = accuracy_score(st.session_state.y_train, st.session_state.y_pred_train)
            test_acc = accuracy_score(st.session_state.y_test, st.session_state.y_pred_test)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Train Accuracy", f"{train_acc:.4f}")
            with col2:
                st.metric("Test Accuracy", f"{test_acc:.4f}")
            
            # Confusion Matrix
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Training Confusion Matrix")
                cm_train = confusion_matrix(st.session_state.y_train, st.session_state.y_pred_train)
                fig = px.imshow(cm_train, text_auto=True,
                              labels=dict(x="Predicted", y="Actual"),
                              title="Training Set Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Test Confusion Matrix")
                cm_test = confusion_matrix(st.session_state.y_test, st.session_state.y_pred_test)
                fig = px.imshow(cm_test, text_auto=True,
                              labels=dict(x="Predicted", y="Actual"),
                              title="Test Set Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
            
            # Classification Report
            st.subheader("Classification Report (Test Set)")
            report = classification_report(st.session_state.y_test, st.session_state.y_pred_test, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            # Advanced Classification Visualizations
            st.markdown("---")
            st.subheader("Advanced Analysis")
            
            class_tabs = st.tabs(["ROC Curves", "Learning Curves", "Feature Importance", "Enhanced Confusion Matrix"])
            
            with class_tabs[0]:
                # ROC Curves (for binary and multiclass)
                n_classes = len(np.unique(st.session_state.y_train))
                
                if n_classes == 2:
                    # Binary classification
                    if hasattr(st.session_state.model, 'predict_proba'):
                        y_score = st.session_state.model.predict_proba(st.session_state.X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(st.session_state.y_test, y_score)
                        roc_auc = auc(fpr, tpr)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            name=f'ROC curve (AUC = {roc_auc:.3f})',
                            mode='lines',
                            line=dict(color='blue', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            name='Random Classifier',
                            mode='lines',
                            line=dict(color='red', dash='dash')
                        ))
                        fig.update_layout(
                            title='Receiver Operating Characteristic (ROC) Curve',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            hovermode='closest'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("AUC Score", f"{roc_auc:.4f}")
                        with col2:
                            if roc_auc >= 0.9:
                                st.success("Excellent classifier!")
                            elif roc_auc >= 0.8:
                                st.info("Good classifier")
                            else:
                                st.warning("Fair classifier")
                    else:
                        st.warning("ROC curve requires probability predictions. This model doesn't support predict_proba()")
                else:
                    # Multiclass ROC
                    if hasattr(st.session_state.model, 'predict_proba'):
                        from sklearn.preprocessing import label_binarize
                        y_test_bin = label_binarize(st.session_state.y_test, classes=np.unique(st.session_state.y_train))
                        y_score = st.session_state.model.predict_proba(st.session_state.X_test)
                        
                        fig = go.Figure()
                        for i in range(n_classes):
                            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                            roc_auc = auc(fpr, tpr)
                            fig.add_trace(go.Scatter(
                                x=fpr, y=tpr,
                                name=f'Class {i} (AUC = {roc_auc:.3f})',
                                mode='lines'
                            ))
                        
                        fig.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            name='Random',
                            mode='lines',
                            line=dict(color='black', dash='dash')
                        ))
                        fig.update_layout(
                            title='ROC Curves - Multiclass',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("ROC curve requires probability predictions")
            
            with class_tabs[1]:
                # Learning Curves
                with st.spinner("Generating learning curves..."):
                    train_sizes = np.linspace(0.1, 1.0, 10)
                    train_sizes_abs, train_scores, test_scores = learning_curve(
                        st.session_state.model, st.session_state.X_train, st.session_state.y_train,
                        train_sizes=train_sizes, cv=5, scoring='accuracy', n_jobs=-1
                    )
                    
                    train_scores_mean = train_scores.mean(axis=1)
                    train_scores_std = train_scores.std(axis=1)
                    test_scores_mean = test_scores.mean(axis=1)
                    test_scores_std = test_scores.std(axis=1)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=train_sizes_abs, y=train_scores_mean,
                        name='Training Score',
                        mode='lines+markers',
                        line=dict(color='blue'),
                        error_y=dict(type='data', array=train_scores_std, visible=True)
                    ))
                    fig.add_trace(go.Scatter(
                        x=train_sizes_abs, y=test_scores_mean,
                        name='Cross-validation Score',
                        mode='lines+markers',
                        line=dict(color='green'),
                        error_y=dict(type='data', array=test_scores_std, visible=True)
                    ))
                    fig.update_layout(
                        title="Learning Curves",
                        xaxis_title="Training Examples",
                        yaxis_title="Accuracy Score",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("Learning curves show model performance vs training size. Converging lines indicate good generalization.")
            
            with class_tabs[2]:
                # Feature Importance
                if hasattr(st.session_state.model, 'coef_'):
                    # For logistic regression with binary classification
                    if n_classes == 2:
                        coef_df = pd.DataFrame({
                            'Feature': st.session_state.feature_names,
                            'Coefficient': st.session_state.model.coef_[0]
                        }).sort_values('Coefficient', key=abs, ascending=False)
                        
                        fig = px.bar(coef_df, x='Feature', y='Coefficient',
                                    title="Feature Coefficients (Logistic Regression)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("Multiclass coefficients:")
                        for i in range(n_classes):
                            st.write(f"**Class {i}:**")
                            coef_df = pd.DataFrame({
                                'Feature': st.session_state.feature_names,
                                'Coefficient': st.session_state.model.coef_[i]
                            }).sort_values('Coefficient', key=abs, ascending=False)
                            st.dataframe(coef_df.head(5))
                elif hasattr(st.session_state.model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Importance': st.session_state.model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='Feature', y='Importance',
                                title="Feature Importance (Tree-based Model)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type")
            
            with class_tabs[3]:
                # Enhanced confusion matrix with percentages
                col1, col2 = st.columns(2)
                
                with col1:
                    cm_train = confusion_matrix(st.session_state.y_train, st.session_state.y_pred_train)
                    cm_train_pct = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis] * 100
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cm_train_pct,
                        x=[f'Pred {i}' for i in range(len(cm_train))],
                        y=[f'Actual {i}' for i in range(len(cm_train))],
                        text=[[f'{cm_train[i,j]}<br>({cm_train_pct[i,j]:.1f}%)' 
                               for j in range(len(cm_train[i]))] for i in range(len(cm_train))],
                        texttemplate='%{text}',
                        colorscale='Blues',
                        showscale=True
                    ))
                    fig.update_layout(title='Training Confusion Matrix (%)')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    cm_test = confusion_matrix(st.session_state.y_test, st.session_state.y_pred_test)
                    cm_test_pct = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis] * 100
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=cm_test_pct,
                        x=[f'Pred {i}' for i in range(len(cm_test))],
                        y=[f'Actual {i}' for i in range(len(cm_test))],
                        text=[[f'{cm_test[i,j]}<br>({cm_test_pct[i,j]:.1f}%)' 
                               for j in range(len(cm_test[i]))] for i in range(len(cm_test))],
                        texttemplate='%{text}',
                        colorscale='Blues',
                        showscale=True
                    ))
                    fig.update_layout(title='Test Confusion Matrix (%)')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Step 5: Simulate Outcomes
        st.markdown('<h2 class="sub-header">5️⃣ Generate Simulated Outcomes</h2>', unsafe_allow_html=True)
        
        n_simulations = st.slider("Number of Simulations", 100, 10000, 1000, 100)
        
        if st.button("🎲 Run Simulation"):
            with st.spinner("Running simulations..."):
                # Generate new synthetic data
                if problem_type == "Regression":
                    X_sim, y_sim_actual = make_regression(
                        n_samples=n_simulations,
                        n_features=n_features,
                        noise=noise,
                        random_state=random_state + 1
                    )
                else:
                    X_sim, y_sim_actual = make_classification(
                        n_samples=n_simulations,
                        n_features=n_features,
                        n_informative=n_informative,
                        n_classes=n_classes,
                        random_state=random_state + 1,
                        flip_y=noise/100
                    )
                
                # Scale if needed
                if scale_features:
                    X_sim_scaled = st.session_state.scaler.transform(X_sim)
                else:
                    X_sim_scaled = X_sim
                
                # Predict
                y_sim_pred = st.session_state.model.predict(X_sim_scaled)
                
                st.success(f"Generated {n_simulations} simulated predictions!")
                
                # Compare distributions
                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=y_sim_actual, name='Actual', opacity=0.7))
                    fig.add_trace(go.Histogram(x=y_sim_pred, name='Predicted', opacity=0.7))
                    fig.update_layout(
                        title="Simulated Data: Actual vs Predicted Distribution",
                        xaxis_title="Value",
                        yaxis_title="Frequency",
                        barmode='overlay'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if problem_type == "Regression":
                        sim_mse = mean_squared_error(y_sim_actual, y_sim_pred)
                        sim_r2 = r2_score(y_sim_actual, y_sim_pred)
                        st.metric("Simulation MSE", f"{sim_mse:.2f}")
                        st.metric("Simulation R²", f"{sim_r2:.4f}")
                    else:
                        sim_acc = accuracy_score(y_sim_actual, y_sim_pred)
                        st.metric("Simulation Accuracy", f"{sim_acc:.4f}")
                    
                    # Statistical comparison
                    st.write("**Statistical Comparison:**")
                    comp_df = pd.DataFrame({
                        'Metric': ['Mean', 'Std Dev', 'Min', 'Max'],
                        'Actual': [y_sim_actual.mean(), y_sim_actual.std(), 
                                  y_sim_actual.min(), y_sim_actual.max()],
                        'Predicted': [y_sim_pred.mean(), y_sim_pred.std(),
                                     y_sim_pred.min(), y_sim_pred.max()]
                    })
                    st.dataframe(comp_df, use_container_width=True)
        
        # Step 6: Model Management & Deployment
        st.markdown('<h2 class="sub-header">6️⃣ Model Management & Deployment</h2>', unsafe_allow_html=True)
        
        mgmt_tabs = st.tabs(["Save/Load Model", "Make Predictions", "Session History", "Export Report"])
        
        with mgmt_tabs[0]:
            st.subheader("Model Persistence")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Save Current Model**")
                model_name = st.text_input("Model Name", value=f"{algorithm.replace(' ', '_')}_{problem_type}")
                
                if st.button("Save Model"):
                    # Create model package
                    model_package = {
                        'model': st.session_state.model,
                        'scaler': st.session_state.scaler if 'scaler' in st.session_state else None,
                        'feature_names': st.session_state.feature_names,
                        'algorithm': algorithm,
                        'problem_type': problem_type,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Serialize to bytes
                    model_bytes = pickle.dumps(model_package)
                    
                    st.download_button(
                        label="Download Model File",
                        data=model_bytes,
                        file_name=f"{model_name}.pkl",
                        mime="application/octet-stream"
                    )
                    st.success("Model ready for download!")
            
            with col2:
                st.write("**Load Saved Model**")
                uploaded_model = st.file_uploader("Upload Model File (.pkl)", type=['pkl'])
                
                if uploaded_model is not None:
                    try:
                        model_package = pickle.loads(uploaded_model.read())
                        st.session_state.model = model_package['model']
                        st.session_state.scaler = model_package['scaler']
                        st.session_state.feature_names = model_package['feature_names']
                        
                        st.success(f"Loaded model: {model_package['algorithm']}")
                        st.info(f"Problem Type: {model_package['problem_type']}")
                        st.info(f"Saved: {model_package['timestamp']}")
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
        
        with mgmt_tabs[1]:
            st.subheader("Prediction Interface")
            st.write("Enter feature values to get predictions from the trained model")
            
            # Create input fields for each feature
            input_data = {}
            cols = st.columns(min(3, n_features))
            
            for idx, feature in enumerate(st.session_state.feature_names):
                with cols[idx % 3]:
                    input_data[feature] = st.number_input(
                        feature,
                        value=0.0,
                        format="%.2f",
                        key=f"pred_{feature}"
                    )
            
            if st.button("Predict"):
                # Prepare input
                X_pred = np.array([list(input_data.values())])
                
                # Scale if needed
                if 'scaler' in st.session_state and st.session_state.scaler is not None:
                    X_pred = st.session_state.scaler.transform(X_pred)
                
                # Make prediction
                prediction = st.session_state.model.predict(X_pred)[0]
                
                st.success(f"**Prediction: {prediction:.4f}**")
                
                # If classification, show probabilities
                if problem_type == "Classification" and hasattr(st.session_state.model, 'predict_proba'):
                    proba = st.session_state.model.predict_proba(X_pred)[0]
                    proba_df = pd.DataFrame({
                        'Class': range(len(proba)),
                        'Probability': proba
                    })
                    fig = px.bar(proba_df, x='Class', y='Probability',
                               title="Class Probabilities")
                    st.plotly_chart(fig, use_container_width=True)
        
        with mgmt_tabs[2]:
            st.subheader("Experiment History")
            
            # Save current experiment
            if st.button("Save Current Experiment"):
                st.session_state.experiment_counter += 1
                experiment = {
                    'id': st.session_state.experiment_counter,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'algorithm': algorithm,
                    'problem_type': problem_type,
                    'n_samples': n_samples,
                    'n_features': n_features,
                    'test_size': test_size,
                }
                
                if problem_type == "Regression":
                    experiment.update({
                        'test_mse': mean_squared_error(st.session_state.y_test, st.session_state.y_pred_test),
                        'test_r2': r2_score(st.session_state.y_test, st.session_state.y_pred_test),
                        'test_mae': mean_absolute_error(st.session_state.y_test, st.session_state.y_pred_test)
                    })
                else:
                    experiment.update({
                        'test_accuracy': accuracy_score(st.session_state.y_test, st.session_state.y_pred_test)
                    })
                
                st.session_state.experiment_history.append(experiment)
                st.success(f"Experiment #{st.session_state.experiment_counter} saved!")
            
            # Display history
            if st.session_state.experiment_history:
                history_df = pd.DataFrame(st.session_state.experiment_history)
                st.dataframe(history_df, use_container_width=True)
                
                # Download history
                csv = history_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download History CSV",
                    data=csv,
                    file_name=f"experiment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No experiments saved yet. Click 'Save Current Experiment' to start tracking.")
        
        with mgmt_tabs[3]:
            st.subheader("Export Analysis Report")
            
            if st.button("Generate HTML Report"):
                # Create HTML report
                html_report = f"""
                <html>
                <head>
                    <title>Model Analysis Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        h1 {{ color: #1f77b4; }}
                        h2 {{ color: #ff7f0e; margin-top: 30px; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #1f77b4; color: white; }}
                        .metric {{ background-color: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                    </style>
                </head>
                <body>
                    <h1>Synthetic Data Modeling Report</h1>
                    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    
                    <h2>Dataset Information</h2>
                    <div class="metric">
                        <p><strong>Problem Type:</strong> {problem_type}</p>
                        <p><strong>Algorithm:</strong> {algorithm}</p>
                        <p><strong>Samples:</strong> {n_samples}</p>
                        <p><strong>Features:</strong> {n_features}</p>
                        <p><strong>Test Size:</strong> {test_size*100:.0f}%</p>
                    </div>
                    
                    <h2>Model Performance</h2>
                """
                
                if problem_type == "Regression":
                    test_mse = mean_squared_error(st.session_state.y_test, st.session_state.y_pred_test)
                    test_r2 = r2_score(st.session_state.y_test, st.session_state.y_pred_test)
                    test_mae = mean_absolute_error(st.session_state.y_test, st.session_state.y_pred_test)
                    
                    html_report += f"""
                    <div class="metric">
                        <p><strong>Test MSE:</strong> {test_mse:.4f}</p>
                        <p><strong>Test R²:</strong> {test_r2:.4f}</p>
                        <p><strong>Test MAE:</strong> {test_mae:.4f}</p>
                    </div>
                    """
                else:
                    test_acc = accuracy_score(st.session_state.y_test, st.session_state.y_pred_test)
                    html_report += f"""
                    <div class="metric">
                        <p><strong>Test Accuracy:</strong> {test_acc:.4f}</p>
                    </div>
                    
                    <h2>Classification Report</h2>
                    """
                    report = classification_report(st.session_state.y_test, st.session_state.y_pred_test, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    html_report += report_df.to_html()
                
                html_report += """
                    <h2>Feature Names</h2>
                    <ul>
                """
                for feature in st.session_state.feature_names:
                    html_report += f"<li>{feature}</li>"
                
                html_report += """
                    </ul>
                </body>
                </html>
                """
                
                st.download_button(
                    label="Download HTML Report",
                    data=html_report,
                    file_name=f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
                st.success("Report generated! Click button above to download.")

else:
    # Welcome screen
    st.info("Configure your synthetic dataset in the sidebar and click **Generate Data** to begin!")
    
    st.markdown("""
    ## Project Overview
    
    This application implements a complete **Synthetic Data Modeling & Simulation** workflow:
    
    1. **Generate Synthetic Dataset** - Create data with known properties
    2. **Exploratory Data Analysis** - Discover data characteristics
    3. **Apply Modeling Technique** - Train machine learning models
    4. **Evaluate Performance** - Assess model accuracy and fit
    5. **Simulate Outcomes** - Generate predictions and compare with known properties
    
    ### Features
    - **Regression & Classification** support
    - Interactive visualizations with Plotly
    - Comprehensive statistical analysis
    - Real-time model training and evaluation
    - Simulation and comparison tools
    
    ### Technologies
    - **Streamlit** for interactive UI
    - **Scikit-learn** for modeling
    - **Pandas & NumPy** for data manipulation
    - **Plotly** for visualizations
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with Streamlit | Synthetic Data Modeling Project</p>
    </div>
""", unsafe_allow_html=True)
