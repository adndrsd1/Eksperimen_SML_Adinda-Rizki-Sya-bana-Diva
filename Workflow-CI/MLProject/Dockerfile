FROM python:3.10-slim

WORKDIR /app

COPY saved_model /app/saved_model

RUN pip install --no-cache-dir \
    mlflow \
    pandas \
    scikit-learn \
    cloudpickle

EXPOSE 5000

CMD ["mlflow", "models", "serve", "-m", "/app/saved_model", "-h", "0.0.0.0", "-p", "5000", "--env-manager", "local"]