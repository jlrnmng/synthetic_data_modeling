# 📊 Synthetic Data Modeling & Simulation

An advanced interactive Streamlit application for exploring modeling and simulation concepts using Python. This project generates synthetic datasets and applies machine learning techniques with comprehensive visualization, analysis, and model management capabilities.

## Project Overview

This project implements a complete data modeling workflow:

1. **Generate Synthetic Dataset** - Create data with known properties using scikit-learn
2. **Exploratory Data Analysis (EDA)** - Discover and visualize data characteristics
3. **Advanced Feature Engineering** - Transform, select, and reduce feature dimensions
4. **Apply Modeling Techniques** - Train multiple regression or classification models
5. **Evaluate Performance** - Assess model accuracy with advanced visualizations
6. **Model Management** - Save, load, and deploy models for predictions
7. **Simulate Outcomes** - Generate predictions and compare with known properties

## Features

### Core Functionality
- **Interactive UI** - Easy-to-use Streamlit interface with dark mode support
- **Dual Model Support** - Regression and Classification tasks
- **Multiple Algorithms** - Linear/Logistic Regression, Random Forest, Decision Tree, SVM
- **Custom Naming** - Name your datasets, features, and target variables

Note: The app now generates datasets that mimic common weather features. The minimal/base schema (used for modeling) includes these core fields:

- `Location`, `MinTemp`, `MaxTemp`, `Rainfall`, `RainTomorrow` (target: 0/1)

Full generated data also includes `Temp9am`, `Temp3pm`, `Humidity9am`, `Humidity3pm`, `WindSpeed9am`, `WindSpeed3pm`, and `RainToday`.

### Data Analysis
### Data Analysis
- **Comprehensive EDA** - Statistical summaries, distributions, correlations, and target analysis
- **Visualizations** - 2D and 3D interactive scatter plots for feature relationships
- **Feature Engineering** - Logarithmic, exponential, square root, and polynomial transforms
- **Feature Selection** - SelectKBest and Recursive Feature Elimination (RFE)
   - **Step 2.5**: Apply feature engineering, selection, PCA, and visualizations (2D & 3D)

### Model Training & Evaluation
- **5 ML Algorithms** - Choose from multiple regression and classification models
- **Real-time Training** - Train models with customizable parameters
- **Advanced Metrics** - MSE, MAE, R², Accuracy, Precision, Recall, F1-Score
- **Learning Curves** - Visualize model performance vs training size
- **ROC Curves & AUC** - For classification models with probability support
- **Residual Analysis** - Detailed residual plots and Q-Q plots for regression
- **Feature Importance** - Visualize coefficients and tree-based feature importances
- **Enhanced Confusion Matrix** - Heatmaps with counts and percentages

### Model Management & Deployment
- **Save/Load Models** - Persist trained models with pickle
- **Prediction Interface** - Make predictions on new data
- **Session History** - Track experiments and compare results
- **Export Reports** - Generate HTML reports with all metrics and analysis
- **CSV Downloads** - Download generated datasets and experiment history

### User Experience
- **Dark Mode** - Toggle between light and dark themes
- **Rich Visualizations** - Interactive Plotly charts and graphs
- **Simulation Tools** - Generate and compare simulated outcomes

## Technologies

- **Streamlit** - Interactive web application framework
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms and tools
- **Matplotlib & Seaborn** - Static visualizations
- **Plotly** - Interactive 3D and 2D visualizations
- **SciPy** - Statistical functions and distributions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jlrnmng/synthetic_data_modeling.git
cd synthetic_data_modeling
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Configure your dataset in the sidebar:
   - Select model type (Regression or Classification)
   - Choose algorithm (5 options available)
   - Adjust dataset parameters (samples, features, noise)
   - Optional: Enable custom naming for dataset and features
   - Click "Generate Data"

4. Explore the comprehensive workflow:
   - **Step 1**: View dataset samples and download CSV
   - **Step 2**: Analyze distributions, correlations, and target variables
   - **Step 2.5**: Apply feature engineering, selection, PCA, and visualizations (2D & 3D)
   - **Step 3**: Train your model with chosen algorithm
   - **Step 4**: Evaluate with advanced metrics and visualizations
   - **Step 5**: Run simulations to test model predictions
   - **Step 6**: Save models, make predictions, track history, export reports

## Algorithms Available

### Regression
- **Linear Regression** - Fast, interpretable baseline
- **Random Forest** - Ensemble method with feature importance
- **Decision Tree** - Non-linear relationships
- **Support Vector Machine** - Kernel-based regression

### Classification
- **Logistic Regression** - Linear decision boundary
- **Random Forest** - Ensemble classifier with high accuracy
- **Decision Tree** - Interpretable tree-based decisions
- **Support Vector Machine** - Maximum margin classifier

## Learning Objectives

This project helps you gain hands-on experience with:
- Synthetic data generation with known properties
- Comprehensive exploratory data analysis
- Advanced feature engineering and selection techniques
- Dimensionality reduction with PCA
- Training multiple ML algorithms
- Model evaluation with various metrics
- ROC curves and learning curves analysis
- Feature importance interpretation
- Model persistence and deployment
- Experiment tracking and reporting

## Advanced Features

### Feature Engineering
- Mathematical transformations (log, exp, sqrt, square)
- Automated feature selection (SelectKBest, RFE)
- PCA with explained variance visualization
- 3D feature space exploration

### Model Analysis
- Learning curves for bias-variance tradeoff
- ROC curves and AUC scores
- Residual analysis and Q-Q plots
- Feature importance for interpretability
- Enhanced confusion matrices with percentages

### Deployment Tools
- Save/load trained models
- Interactive prediction interface
- Session history tracking
- Automated HTML report generation
- CSV export for all data

## Project Structure

```
synthetic_data_modeling/
├── app.py                 # Main Streamlit application
├── test_app.py           # Comprehensive test suite
├── requirements.txt       # Python dependencies
├── project_details.txt    # Project specifications
├── README.md             # This file
└── script/               # Additional scripts
    └── test_app.py       # Test script backup
```

##  Testing

Run the comprehensive test suite to verify all functionalities:

```bash
python test_app.py
```

The test script covers:
- Data generation (regression & classification)
- Feature engineering transformations
- Feature selection methods (SelectKBest, RFE)
- Dimensionality reduction (PCA)
- All 5 ML algorithms for both problem types
- Model persistence (save/load)
- Custom naming functionality
- Statistical analysis
- Prediction interface

Expected output: All 10 tests should pass with ✓ marks.

##  Troubleshooting

**Issue: Streamlit won't start**
```bash
# Try using python.exe explicitly
python.exe -m streamlit run app.py
```

**Issue: Missing dependencies**
```bash
# Reinstall all requirements
pip install -r requirements.txt --upgrade
```

**Issue: Dark mode not working**
Dark-mode toggle was removed; theme is controlled by the app's CSS. Refresh the browser if needed.

**Issue: Models not saving**
- Ensure you have write permissions in the directory
- Check available disk space

##  Author

Created as part of a modeling and simulation learning project.
- **Group Members**: [Garcia, Asi Neo] [Manaog, Johnlerein B.] [Valera, Joshua Ean]
- **Repository**: [synthetic_data_modeling](https://github.com/jlrnmng/synthetic_data_modeling)
- **GitHub**: [@jlrnmng](https://github.com/jlrnmng)

##  Project Statistics

- **Lines of Code**: ~1,400+
- **Features**: 15+ major features
- **Algorithms**: 5 ML algorithms (Linear/Logistic Regression, Random Forest, Decision Tree, SVM)
- **Test Coverage**: 10 comprehensive tests
- **Visualizations**: 20+ interactive charts and plots

##  Use Cases

This application is ideal for:
- **Educational Purposes** - Learn ML concepts interactively
- **Data Science Projects** - Quick prototyping and experimentation
- **Model Comparison** - Test different algorithms on synthetic data
- **Feature Engineering Studies** - Explore transformation effects
- **Finals Projects** - Comprehensive ML platform demonstration
- **Research** - Generate controlled datasets for testing hypotheses
