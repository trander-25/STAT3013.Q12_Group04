# TELECOM CHURN PREDICTION WITH STATISTICAL AND MACHINE LEARNING MODELS

**STAT3013.Q12 - Group 04 (BaXin)**

A machine learning project for predicting customer churn in telecommunications using various classification algorithms including LightGBM, XGBoost, CatBoost, Random Forest, and Logistic Regression.

## Project Overview

This project analyzes customer behavior data to predict churn (customer attrition) using multiple machine learning models. The dataset includes customer usage patterns, demographics, and service plan information. The project includes comprehensive data preprocessing, feature engineering, model training, and evaluation.

## Dataset & Resources

### Dataset
- **Training Data**: `./Train.csv`
- **Dataset Link**: https://zindi.africa/competitions/expresso-churn-prediction/data

### Video Demo
- **Video Link**: https://drive.google.com/file/d/1MR10iq9CHQhqACQ9EHfCfQQFVwTNDeBR/view?usp=sharing

## Requirements

### Python Version
- Python 3.8 or higher

### Required Libraries
```
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
xgboost
catboost
imbalanced-learn
jupyter
```

### Installation

Install all required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost imbalanced-learn jupyter
```

Or install from a requirements file (if available):

```bash
pip install -r requirements.txt
```

## Dataset

The project expects a dataset file at `./Train.csv` containing customer information including:
- Customer usage metrics (MONTANT, FREQUENCE, etc.)
- Service plan information (TOP_PACK)
- Demographic data (REGION)
- Target variable (CHURN)

## Project Structure

```
STAT3013.Q12_Group04/
│
├── descriptive statistic.ipynb    # Exploratory data analysis
├── g4_lightgbm.ipynb              # LightGBM model implementation
├── g4-catboost.ipynb              # CatBoost model implementation
├── g4-logistic-regression.ipynb   # Logistic Regression model
├── g4-random-forest.ipynb         # Random Forest model implementation
├── g4-xgboost.ipynb               # XGBoost model implementation
└── README.md                      
```

## Run Instructions

### 1. Data Exploration

Start with the descriptive statistics notebook to understand the dataset:

```bash
jupyter notebook "descriptive statistic.ipynb"
```

### 2. Model Training

Run any of the model notebooks to train and evaluate specific algorithms:

**LightGBM Model:**
```bash
jupyter notebook g4_lightgbm.ipynb
```

**XGBoost Model:**
```bash
jupyter notebook g4-xgboost.ipynb
```

**CatBoost Model:**
```bash
jupyter notebook g4-catboost.ipynb
```

**Random Forest Model:**
```bash
jupyter notebook g4-random-forest.ipynb
```

**Logistic Regression Model:**
```bash
jupyter notebook g4-logistic-regression.ipynb
```

### 3. Workflow

Each model notebook follows a similar workflow:

1. **Data Loading**: Load the training dataset from `./Train.csv`
2. **Data Cleaning**: Remove inactive churners (customers with zero usage)
3. **Feature Engineering**: 
   - Clean and standardize TOP_PACK column
   - Create pack groups
   - Encode categorical variables
4. **Model Training**: Train the specific classifier with optimized parameters
5. **Evaluation**: 
   - Calculate metrics (accuracy, precision, recall, F1-score, AUC-ROC)
   - Generate confusion matrix
   - Plot ROC curve
   - Analyze feature importance
6. **Threshold Optimization**: Find optimal classification threshold

### 4. Running in VS Code

If using VS Code with Jupyter extension:

1. Open any `.ipynb` file
2. Select Python kernel
3. Run cells sequentially using `Shift + Enter` or "Run All"

## Key Features

- **Data Preprocessing**: Handling missing values, removing inactive churners, feature standardization
- **Feature Engineering**: TOP_PACK cleaning, pack grouping, categorical encoding
- **Multiple Models**: Comparison of various algorithms (tree-based and linear models)
- **Model Evaluation**: Comprehensive metrics including confusion matrix, ROC curves, and feature importance
- **Threshold Optimization**: Finding optimal decision thresholds for business objectives

## Results

Each notebook provides:
- Model performance metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
- Confusion matrices
- ROC curves
- Feature importance rankings
- Optimal classification thresholds

## License

MIT License

Copyright (c) 2025 STAT3013.Q12_Group04

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

This is an academic project for STAT3013.Q12. For questions or suggestions, please contact the team members.

## Acknowledgments

- Course: STAT3013.Q12
- Institution: University of Information Technology, VNUHCM
- Dataset: Customer churn data for telecommunications analysis
