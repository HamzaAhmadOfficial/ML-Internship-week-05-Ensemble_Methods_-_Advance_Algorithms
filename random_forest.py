# random_forest.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

#  Load dataset
df = pd.read_csv("data/heart.csv")

#  Split features & target
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#  Base Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("Base RF Accuracy:",
      accuracy_score(y_test, rf.predict(X_test)))

print("OOB Score:", rf.oob_score_)

#  GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [3, 5, 10, None]
}

grid = GridSearchCV(
    RandomForestClassifier(oob_score=True, random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

best_rf = grid.best_estimator_

print("Best RF Accuracy:",
      accuracy_score(y_test, best_rf.predict(X_test)))

#  Compare with Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

print("Decision Tree Accuracy:",
      accuracy_score(y_test, dt.predict(X_test)))

#  Feature Importance Plot
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)),
           X.columns[indices],
           rotation=90)
plt.tight_layout()
plt.savefig("outputs/rf_feature_importance.png")
plt.show()

#  OOB Error vs n_estimators
oob_errors = []
n_range = [10, 50, 100, 200]

for n in n_range:
    model = RandomForestClassifier(
        n_estimators=n,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    oob_errors.append(1 - model.oob_score_)

plt.figure()
plt.plot(n_range, oob_errors, marker='o')
plt.xlabel("n_estimators")
plt.ylabel("OOB Error")
plt.title("OOB Error vs n_estimators")
plt.savefig("outputs/rf_oob_error.png")
plt.show()

#  Save model
joblib.dump(best_rf, "models/best_random_forest.pkl")

print("Random Forest model saved successfully.")
