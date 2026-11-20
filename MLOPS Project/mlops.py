
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

# ------------------------------------------------
# 2. Split Dataset
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------------
# 3. MLflow Experiment
# ------------------------------------------------

mlflow.set_experiment("iris_Classification_Experiment")

# ------------------------------------------------
# 4. Logistic Regression Model
# ------------------------------------------------

with mlflow.start_run(run_name="Logistic_Regression"):

    lr_model = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=1000
    )

    lr_model.fit(X_train, y_train)

    lr_prediction = lr_model.predict(X_test)

    lr_accuracy = accuracy_score(y_test, lr_prediction)

    lr_confusion_matrix = confusion_matrix(
        y_test,
        lr_prediction
    )

    # Log parameters and metrics
    mlflow.log_param("model", "Logistic_Regression")
    mlflow.log_param("solver", "liblinear")
    mlflow.log_param("C", 1000)

    mlflow.log_metric("accuracy", lr_accuracy)

    # Confusion Matrix Visualization
    plt.figure(figsize=(5, 5))

    sns.heatmap(
        lr_confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )

    plt.title("Confusion Matrix - Logistic Regression")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("lr_confusion_matrix.png")
    plt.close()

    # Log artifact
    mlflow.log_artifact("lr_confusion_matrix.png")

    # Log model
    mlflow.sklearn.log_model(
        lr_model,
        "Logistic_Regression_model"
    )

    # Register model
    log_model_uri = (
        f"runs:/{mlflow.active_run().info.run_id}/"
        "Logistic_Regression_model"
    )

    registered_lr_model = mlflow.register_model(
        log_model_uri,
        "Logistic_Regression_model"
    )

    print("Logistic Regression Accuracy:", lr_accuracy)
    print("Confusion Matrix:")
    print(lr_confusion_matrix)


# ------------------------------------------------
# 5. Random Forest Model

print("-----Start Random Forest------")
# ------------------------------------------------


# ------------------------------------------------
# 5. Random Forest Model
# ------------------------------------------------

print("\nStarting Random Forest Model Training...")

with mlflow.start_run(run_name="Random_Forest"):

    # Initialize Model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        random_state=0,
        criterion="entropy"
    )

    # Train Model
    rf_model.fit(X_train, y_train)

    print("Random Forest Model Training Completed!")

    # Prediction
    rf_prediction = rf_model.predict(X_test)

    # Evaluation
    rf_accuracy = accuracy_score(y_test, rf_prediction)

    rf_confusion_matrix = confusion_matrix(
        y_test,
        rf_prediction
    )

    # MLflow Parameters
    mlflow.log_param("model", "Random_Forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("criterion", "entropy")

    # MLflow Metrics
    mlflow.log_metric("accuracy", rf_accuracy)

    # Confusion Matrix
    plt.figure(figsize=(5, 5))

    sns.heatmap(
        rf_confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )

    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("rf_confusion_matrix.png")
    plt.close()

    # Log Artifact
    mlflow.log_artifact("rf_confusion_matrix.png")

    # Log Model
    mlflow.sklearn.log_model(
        rf_model,
        "random_forest_model"
    )

    # Register Model
    log_model_uri = (
        f"runs:/{mlflow.active_run().info.run_id}/"
        "random_forest_model"
    )

    registered_rf_model = mlflow.register_model(
        log_model_uri,
        "Random_Forest_model"
    )

    print("\nRandom Forest Accuracy:", rf_accuracy)

    print("Random Forest Confusion Matrix:")
    print(rf_confusion_matrix)

    print("Random Forest Model Registered Successfully!")

print("\n========== RANDOM FOREST COMPLETED ==========")


# ------------------------------------------------
# 6. Load Registered Models
# ------------------------------------------------

logistic_model_uri = (
    f"models:/Logistic_Regression_model/"
    f"{registered_lr_model.version}"
)

loaded_lr_model = mlflow.sklearn.load_model(
    logistic_model_uri
)

random_forest_model_uri = (
    f"models:/Random_Forest_model/"
    f"{registered_rf_model.version}"
)

loaded_rf_model = mlflow.sklearn.load_model(
    random_forest_model_uri
)

print("\nBoth models loaded successfully!")

print("Logistic Regression Model:", loaded_lr_model)
print("Random Forest Model:", loaded_rf_model)
