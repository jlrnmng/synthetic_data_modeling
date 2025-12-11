# 📊 Synthetic Data Modeling & Simulation

An interactive Streamlit application for exploring modeling and simulation concepts using Python. This project generates synthetic datasets and applies machine learning techniques with comprehensive visualization and analysis.

## 🎯 Project Overview

This project implements a complete data modeling workflow:

1. **Generate Synthetic Dataset** - Create data with known properties using scikit-learn
2. **Exploratory Data Analysis (EDA)** - Discover and visualize data characteristics
3. **Apply Modeling Techniques** - Train regression or classification models
4. **Evaluate Performance** - Assess model accuracy and fit quality
5. **Simulate Outcomes** - Generate predictions and compare with known properties

## ✨ Features

- **Interactive UI** - Easy-to-use Streamlit interface
- **Dual Model Support** - Regression and Classification tasks
- **Comprehensive EDA** - Statistical summaries, distributions, correlations, and target analysis
- **Real-time Training** - Train models with customizable parameters
- **Rich Visualizations** - Interactive Plotly charts and graphs
- **Simulation Tools** - Generate and compare simulated outcomes
- **Model Evaluation** - Detailed metrics, confusion matrices, and residual analysis

## 🛠️ Technologies

- **Streamlit** - Interactive web application framework
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib & Seaborn** - Static visualizations
- **Plotly** - Interactive visualizations

## 📦 Installation

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

## 🚀 Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Configure your dataset in the sidebar:
   - Select model type (Regression or Classification)
   - Adjust dataset parameters (samples, features, noise)
   - Click "Generate Data"

4. Explore the workflow:
   - View dataset samples and statistics
   - Analyze distributions and correlations
   - Train your model
   - Evaluate performance
   - Run simulations

## 📊 Model Types

### Regression
- Linear regression modeling
- R² and MSE metrics
- Residual analysis
- Feature coefficient analysis

### Classification
- Logistic regression
- Accuracy metrics
- Confusion matrices
- Classification reports

## 🎓 Learning Objectives

This project helps you gain hands-on experience with:
- Synthetic data generation
- Exploratory data analysis techniques
- Feature engineering concepts
- Model training and evaluation
- Statistical analysis and visualization
- Model simulation and validation

## 📝 Project Structure

```
synthetic_data_modeling/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── project_details.txt    # Project specifications
└── README.md             # This file
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

Created as part of a modeling and simulation learning project.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Scikit-learn](https://scikit-learn.org/)
- Visualizations by [Plotly](https://plotly.com/)
