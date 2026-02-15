"""
Task 5.4 - Model Evaluation & Cross Validation

Implements:
- K-Fold Cross Validation
- Stratified K-Fold
- Leave-One-Out CV
- Confidence Intervals
- Learning Curves
- Validation Curves
- Plot Saving (outputs folder)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    learning_curve,
    validation_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Create Output Folder

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load Dataset

data = load_breast_cancer()
X, y = data.data, data.target

print("Dataset shape:", X.shape)


# Define Models

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=500))
    ]),

    "Random Forest": RandomForestClassifier(n_estimators=100),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf"))
    ])
}


# Confidence Interval Function

def confidence_interval(scores, confidence=0.95):
    mean = np.mean(scores)
    std = np.std(scores)
    n = len(scores)

    margin = 1.96 * (std / np.sqrt(n))
    return mean, std, (mean - margin, mean + margin)


# K-Fold Cross Validation

print("\n===== K-Fold CV (k=5) =====")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kfold)

    mean, std, ci = confidence_interval(scores)

    print(f"\n{name}")
    print("Scores:", scores)
    print(f"Mean Accuracy: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print(f"95% CI: {ci}")


# Stratified K-Fold

print("\n===== Stratified K-Fold CV =====")

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=skfold)

    mean, std, ci = confidence_interval(scores)

    print(f"\n{name}")
    print(f"Mean Accuracy: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print(f"95% CI: {ci}")


# Leave-One-Out CV

print("\n===== Leave-One-Out CV =====")

loo = LeaveOneOut()

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=loo, n_jobs=-1)

    print(f"\n{name}")
    print("Mean Accuracy:", scores.mean())


# Learning Curve Function

def plot_learning_curve(model, title):

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, label="Training Score")
    plt.plot(train_sizes, val_mean, label="Validation Score")

    plt.title(f"Learning Curve - {title}")
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Save plot
    filename = f"{OUTPUT_DIR}/learning_curve_{title.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()


# Generate Learning Curves

print("\nGenerating Learning Curves...")

for name, model in models.items():
    plot_learning_curve(model, name)


# Validation Curve (SVM Hyperparameter)
print("\nGenerating Validation Curve...")

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC())
])

param_range = np.logspace(-3, 2, 6)

train_scores, val_scores = validation_curve(
    svm_model,
    X,
    y,
    param_name="model__C",
    param_range=param_range,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure()
plt.semilogx(param_range, train_mean, label="Training Score")
plt.semilogx(param_range, val_mean, label="Validation Score")

plt.title("Validation Curve (SVM - C Parameter)")
plt.xlabel("C Value")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Save plot
plt.savefig(f"{OUTPUT_DIR}/validation_curve_svm.png",
            dpi=300,
            bbox_inches="tight")

plt.show()
plt.close()


print("\nEvaluation Complete!")
