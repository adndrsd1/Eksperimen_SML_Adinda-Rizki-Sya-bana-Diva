import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("Titanic Survical Prediction")

df = pd.read_csv('../Titanic-Dataset_preprocessing.csv')

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()

with mlflow.start_run():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")

print("Training selesai")