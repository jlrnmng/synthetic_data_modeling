import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Synthetic Data Modeling", layout="wide", page_icon="📊")

# Custom CSS
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
st.markdown('<h1 class="main-header">📊 Synthetic Data Modeling & Simulation</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Regression", "Classification"]
)

# Dataset parameters
st.sidebar.subheader("Dataset Parameters")
n_samples = st.sidebar.slider("Number of Samples", 100, 5000, 1000, 100)
n_features = st.sidebar.slider("Number of Features", 2, 20, 5, 1)
noise = st.sidebar.slider("Noise Level", 0.0, 50.0, 10.0, 1.0)
random_state = st.sidebar.number_input("Random State", 0, 1000, 42)

if model_type == "Classification":
    n_classes = st.sidebar.slider("Number of Classes", 2, 5, 2, 1)
    n_informative = st.sidebar.slider("Informative Features", 2, n_features, min(3, n_features), 1)

# Generate Data Button
generate_data = st.sidebar.button("🎲 Generate Data", type="primary")

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Main content
if generate_data or st.session_state.data_generated:
    st.session_state.data_generated = True
    
    # Step 1: Generate Synthetic Dataset
    st.markdown('<h2 class="sub-header">1️⃣ Generate Synthetic Dataset</h2>', unsafe_allow_html=True)
    
    with st.spinner("Generating synthetic data..."):
        if model_type == "Regression":
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=noise,
                random_state=random_state
            )
            problem_type = "Regression"
        else:
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_classes=n_classes,
                random_state=random_state,
                flip_y=noise/100
            )
            problem_type = "Classification"
        
        # Create DataFrame
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['Target'] = y
        
        # Store in session state
        st.session_state.df = df
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.feature_names = feature_names
        st.session_state.problem_type = problem_type
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", n_samples)
    with col2:
        st.metric("Features", n_features)
    with col3:
        st.metric("Problem Type", problem_type)
    
    st.success("✅ Data generated successfully!")
    
    with st.expander("📋 View Dataset Sample"):
        st.dataframe(st.session_state.df.head(20), use_container_width=True)
    
    # Step 2: Exploratory Data Analysis (EDA)
    st.markdown('<h2 class="sub-header">2️⃣ Exploratory Data Analysis (EDA)</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistical Summary", "📈 Distributions", "🔗 Correlations", "🎯 Target Analysis"])
    
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
        target_corr = corr_matrix['Target'].drop('Target').sort_values(ascending=False)
        st.subheader("Top Features Correlated with Target")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Positive Correlations:**")
            st.dataframe(target_corr.head(5))
        with col2:
            st.write("**Negative Correlations:**")
            st.dataframe(target_corr.tail(5))
    
    with tab4:
        st.subheader("Target Variable Analysis")
        if problem_type == "Classification":
            fig = px.histogram(st.session_state.df, x='Target', 
                             title="Target Class Distribution",
                             color='Target')
            st.plotly_chart(fig, use_container_width=True)
            
            # Class distribution
            class_dist = st.session_state.df['Target'].value_counts()
            st.write("**Class Distribution:**")
            st.dataframe(class_dist)
        else:
            fig = px.histogram(st.session_state.df, x='Target', nbins=50,
                             title="Target Variable Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean", f"{st.session_state.y.mean():.2f}")
                st.metric("Std Dev", f"{st.session_state.y.std():.2f}")
            with col2:
                st.metric("Min", f"{st.session_state.y.min():.2f}")
                st.metric("Max", f"{st.session_state.y.max():.2f}")
    
    # Step 3: Apply Modeling Technique
    st.markdown('<h2 class="sub-header">3️⃣ Apply Modeling Technique</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 50, 20, 5) / 100
    with col2:
        scale_features = st.checkbox("Scale Features", value=True)
    
    train_model = st.button("🚀 Train Model", type="primary")
    
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
            if problem_type == "Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                st.session_state.model = model
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.y_pred_train = y_pred_train
                st.session_state.y_pred_test = y_pred_test
            else:
                model = LogisticRegression(max_iter=1000, random_state=random_state)
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                st.session_state.model = model
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.y_pred_train = y_pred_train
                st.session_state.y_pred_test = y_pred_test
        
        st.success("✅ Model trained successfully!")
        
        # Step 4: Model Evaluation
        st.markdown('<h2 class="sub-header">4️⃣ Model Evaluation</h2>', unsafe_allow_html=True)
        
        if problem_type == "Regression":
            # Metrics
            train_mse = mean_squared_error(st.session_state.y_train, st.session_state.y_pred_train)
            test_mse = mean_squared_error(st.session_state.y_test, st.session_state.y_pred_test)
            train_r2 = r2_score(st.session_state.y_train, st.session_state.y_pred_train)
            test_r2 = r2_score(st.session_state.y_test, st.session_state.y_pred_test)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Train MSE", f"{train_mse:.2f}")
            with col2:
                st.metric("Test MSE", f"{test_mse:.2f}")
            with col3:
                st.metric("Train R²", f"{train_r2:.4f}")
            with col4:
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
            
            # Feature importance
            st.subheader("Feature Coefficients")
            coef_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Coefficient': st.session_state.model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            fig = px.bar(coef_df, x='Feature', y='Coefficient',
                        title="Feature Importance (Coefficients)")
            st.plotly_chart(fig, use_container_width=True)
            
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
        
        # Step 5: Simulate Outcomes
        st.markdown('<h2 class="sub-header">5️⃣ Generate Simulated Outcomes</h2>', unsafe_allow_html=True)
        
        n_simulations = st.slider("Number of Simulations", 100, 5000, 1000, 100)
        
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
                
                st.success(f"✅ Generated {n_simulations} simulated predictions!")
                
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

else:
    # Welcome screen
    st.info("👈 Configure your synthetic dataset in the sidebar and click **Generate Data** to begin!")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This application implements a complete **Synthetic Data Modeling & Simulation** workflow:
    
    1. **Generate Synthetic Dataset** - Create data with known properties
    2. **Exploratory Data Analysis** - Discover data characteristics
    3. **Apply Modeling Technique** - Train machine learning models
    4. **Evaluate Performance** - Assess model accuracy and fit
    5. **Simulate Outcomes** - Generate predictions and compare with known properties
    
    ### 📚 Features
    - **Regression & Classification** support
    - Interactive visualizations with Plotly
    - Comprehensive statistical analysis
    - Real-time model training and evaluation
    - Simulation and comparison tools
    
    ### 🛠️ Technologies
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
