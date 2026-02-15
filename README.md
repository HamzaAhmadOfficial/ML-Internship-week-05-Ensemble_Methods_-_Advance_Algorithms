# Week 5: Ensemble Methods & Advanced Algorithms

This repository contains implementations of advanced machine learning algorithms including Random Forest, XGBoost (Gradient Boosting), Support Vector Machines (SVM), and comprehensive Model Evaluation techniques. These tasks were completed as part of my internship to strengthen understanding of ensemble methods, hyperparameter tuning, and model evaluation strategies.

## Repository Structure

Week_5_Ensemble_Advanced/
- random_forest.py - Random Forest implementation with GridSearchCV and OOB error analysis
- xgboost_implementation.py - XGBoost implementation with SHAP visualization
- svm_classification.py - SVM implementation with different kernels and decision boundary plots
- model_evaluation.py - Cross-validation, learning curves, and validation curves
- data/ - Dataset files
- requirements.txt - Required Python libraries
- README.md - Project overview

## Requirements

Install the required Python packages:
pip install -r requirements.txt

Key libraries used:
- scikit-learn
- xgboost
- matplotlib
- seaborn
- pandas
- numpy
- shap

## Task Overview

### Task 5.1: Random Forest Classifier
- Implemented Random Forest on a classification dataset
- Hyperparameter tuning using GridSearchCV (n_estimators, max_depth)
- Compared performance with a single Decision Tree
- Plotted feature importances
- Analyzed Out-of-Bag (OOB) error
- Saved the best-performing model

Experiments included:
- n_estimators: 10, 50, 100, 200
- max_depth: 3, 5, 10, None
- Plots: OOB error vs n_estimators, feature importance comparison

### Task 5.2: Gradient Boosting (XGBoost)
- Implemented XGBoost classifier for classification tasks
- Hyperparameter tuning: learning_rate, max_depth, n_estimators via GridSearchCV
- Plotted training history using evaluation sets
- Extracted and visualized feature importance
- Created SHAP values summary plot for model interpretability
- Compared XGBoost performance with Random Forest
- Saved models in .pkl and .json formats

### Task 5.3: Support Vector Machines (SVM)
- Implemented SVM with different kernels: linear, rbf, poly
- Visualized decision boundaries on 2D datasets
- Hyperparameter tuning experiments:
  - C: 0.1, 1, 10, 100
  - gamma: 0.001, 0.01, 0.1, 1
- Plotted accuracy vs C and gamma
- Created comparison table for kernel performances
- Saved best-performing SVM model

### Task 5.4: Model Evaluation & Cross-Validation
- Implemented k-fold cross-validation, Stratified K-Fold, and Leave-One-Out CV
- Evaluated multiple models: Logistic Regression, Random Forest, SVM
- Calculated mean, standard deviation, and confidence intervals of CV scores
- Generated learning curves and validation curves for key hyperparameters

## Usage Instructions

1. Install dependencies:
   pip install -r requirements.txt

2. Run scripts individually:
   python random_forest.py
   python xgboost_implementation.py
   python svm_classification.py
   python model_evaluation.py



## References
- Scikit-learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- SHAP Documentation: https://shap.readthedocs.io/
- UCI Machine Learning Repository

## Author
Hamza Ahmad
Internship Project – Week 5: Ensemble Methods & Advanced Algorithms