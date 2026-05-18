import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import dagshub
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# DagsHub login
dagshub.init(
    repo_owner='adndrsd1',
    repo_name='titanic-mlops',
    mlflow=True
)

mlflow.set_experiment(
    "Titanic Survival Prediction"
)

# Load dataset
df = pd.read_csv('../Titanic-Dataset_preprocessing.csv')

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10],
    "min_samples_split": [2, 5]
}

rf = RandomForestClassifier()

grid = GridSearchCV(
    rf,
    params,
    cv=3,
    scoring="accuracy"
)

with mlflow.start_run():

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    # manual logging
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)

    # classification report
    report = classification_report(y_test, preds)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    # confusion matrix
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

    # model logging
    mlflow.sklearn.log_model(
        best_model,
        "model"
    )

    joblib.dump(best_model, "model.pkl")

print("Model lokal berhasil disimpan")

print("Tuning selesai")