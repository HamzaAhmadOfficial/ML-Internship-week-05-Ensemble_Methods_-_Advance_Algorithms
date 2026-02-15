# svm_classification.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

# Load Dataset

df = pd.read_csv("data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]


#  Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


#  Feature Scaling (MANDATORY for SVM)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#  Train Different Kernels

kernels = ["linear", "rbf", "poly"]

results = {}
best_model = None
best_acc = 0

for kernel in kernels:
    model = SVC(kernel=kernel, C=1, gamma="scale")
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)

    results[kernel] = acc
    print(f"{kernel} kernel accuracy:", acc)

    if acc > best_acc:
        best_acc = acc
        best_model = model

print("\nBest Kernel Accuracy:", best_acc)


#  Classification Report

best_preds = best_model.predict(X_test_scaled)

print("\nClassification Report:\n")
print(classification_report(y_test, best_preds))


#  Decision Boundary Visualization
# (Using PCA → reduce to 2D)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

viz_model = SVC(kernel="rbf")
viz_model.fit(X_pca, y_train)

# Mesh grid
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

Z = viz_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_train)
plt.title("SVM Decision Boundary (PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.savefig("outputs/svm_decision_boundary.png")
plt.show()

#  Save Best Model + Scaler

joblib.dump(best_model, "models/best_svm.pkl")
joblib.dump(scaler, "models/svm_scaler.pkl")

print("\nSVM model saved successfully.")
