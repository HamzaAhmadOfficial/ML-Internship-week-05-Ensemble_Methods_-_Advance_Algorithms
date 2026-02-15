# xgboost_implementation.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import shap
import xgboost as xgb

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


# Load Dataset

df = pd.read_csv("data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Base XGBoost Model

model = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

preds = model.predict(X_test)

print("Base XGBoost Accuracy:",
      accuracy_score(y_test, preds))


#  Hyperparameter Tuning (GridSearchCV)

param_grid = {
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "n_estimators": [50, 100, 200]
}

grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

best_xgb = grid.best_estimator_


#  Evaluate Best Model
best_preds = best_xgb.predict(X_test)

print("\nBest XGBoost Accuracy:",
      accuracy_score(y_test, best_preds))

print("\nClassification Report:\n",
      classification_report(y_test, best_preds))


#  Plot Training History

results = model.evals_result()

plt.figure()
plt.plot(results['validation_0']['logloss'], label="Train")
plt.plot(results['validation_1']['logloss'], label="Test")
plt.xlabel("Iterations")
plt.ylabel("Log Loss")
plt.title("XGBoost Training History")
plt.legend()
plt.savefig("outputs/xgb_training_history.png")
plt.show()


# Feature Importance Plot

xgb.plot_importance(best_xgb, max_num_features=10)
plt.title("XGBoost Feature Importance")
plt.savefig("outputs/xgb_feature_importance.png")
plt.show()


#  SHAP Explainability

explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("outputs/shap_summary.png")
plt.show()


#  Save Model

joblib.dump(best_xgb, "models/best_xgboost.pkl")
best_xgb.save_model("models/best_xgboost.json")

print("\nXGBoost model saved successfully.")
