# Titanic Survival Prediction - Kaggle Competition

This repository contains a comprehensive machine learning solution for the famous Titanic survival prediction competition on Kaggle. The project explores various machine learning algorithms and ensemble methods to predict passenger survival on the Titanic.

## 📊 Dataset Overview

The Titanic dataset contains information about passengers aboard the Titanic, including:
- **PassengerId**: Unique identifier for each passenger
- **Survived**: Target variable (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Name**: Passenger name
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## 🚀 Features

### Data Preprocessing
- **Missing Value Handling**: Filled missing Age with mean, Cabin with 'Unknown', and Embarked with 'S'
- **Feature Engineering**: Created new features like family size and age groups
- **Categorical Encoding**: Applied one-hot encoding for categorical variables

### Machine Learning Models Implemented

1. **Random Forest Classifier**
   - Baseline model with 100 estimators
   - Feature importance analysis
   - Cross-validation evaluation

2. **XGBoost Classifier**
   - Hyperparameter tuning with GridSearchCV
   - Improved performance over Random Forest
   - Feature importance visualization

3. **Ensemble Methods**
   - **Voting Classifier**: Hard and soft voting approaches
   - **Stacking Classifier**: Meta-learning with Logistic Regression as final estimator
   - **Weighted Ensemble**: Performance-based weighting of individual models

4. **TensorFlow Decision Forests**
   - Random Forest implementation
   - Gradient Boosted Trees
   - CART (Classification and Regression Trees)

## 📈 Performance Results

The best performing model was **TensorFlow Decision Forests** with cross-validation accuracy of **0.891304** (89.13%) on the validation set.

### Model Comparison
- Random Forest: ~0.82-0.85 accuracy
- XGBoost: ~0.84-0.87 accuracy  
- Stacking Classifier: 0.84916 accuracy
- **TensorFlow Decision Forests: 0.891304 accuracy** (Best) 🏆

## 🛠️ Technical Stack

- **Python 3.9**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting framework
- **TensorFlow Decision Forests**: Advanced tree-based models
- **Matplotlib/Seaborn**: Data visualization

## 📁 Project Structure

```
kaggle-titanic-predictions/
├── titanic.ipynb              # Main Jupyter notebook with complete analysis
├── train.csv                  # Training dataset
├── test.csv                   # Test dataset
├── venv/                      # Python virtual environment
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Jupyter Notebook
- Required packages (see installation below)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd kaggle-titanic-predictions
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook**
   ```bash
   jupyter notebook titanic.ipynb
   ```

## 📊 Key Insights

### Feature Importance
The most important features for survival prediction were:
1. **Sex**: Gender was the strongest predictor
2. **Fare**: Higher fare correlated with survival
3. **Age**: Younger passengers had higher survival rates
4. **Pclass**: First class passengers had better survival rates

### Data Patterns
- **Gender Gap**: Women had significantly higher survival rates than men
- **Class Effect**: First class passengers had better survival rates
- **Age Effect**: Children and elderly had different survival patterns
- **Family Size**: Passengers with families had varying survival rates

## 🎯 Model Selection Strategy

1. **Baseline**: Started with Random Forest for interpretability
2. **Improvement**: Applied XGBoost with hyperparameter tuning
3. **Ensemble**: Combined multiple models using stacking
4. **Advanced**: Explored TensorFlow Decision Forests
5. **Final**: Selected TensorFlow Decision Forests as the best performer

## 📝 Submission Files

Multiple submission files were generated for comparison:
- `submission.csv`: Basic Random Forest predictions
- `submission_xgboost.csv`: XGBoost predictions
- `submission_stacking.csv`: Stacking Classifier predictions
- `submission_tfDT.csv`: TensorFlow Decision Forests predictions (Best) 🏆

## 🤝 Contributing

Feel free to contribute to this project by:
- Improving the models
- Adding new feature engineering techniques
- Optimizing hyperparameters
- Adding new visualization techniques

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Kaggle for hosting the competition
- The Titanic dataset contributors
- The open-source machine learning community

---

**Note**: This project is for educational purposes and demonstrates various machine learning techniques applied to a classic classification problem.
