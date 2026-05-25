from prometheus_client import start_http_server
from prometheus_client import Counter, Histogram, Gauge
import requests
import time
import random

# METRICS

REQUEST_COUNT = Counter(
    'model_requests_total',
    'Total prediction requests'
)

FAILED_REQUESTS = Counter(
    'failed_requests_total',
    'Total failed requests'
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency'
)

CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'CPU usage percent'
)

MEMORY_USAGE = Gauge(
    'memory_usage_percent',
    'Memory usage percent'
)

ACTIVE_USERS = Gauge(
    'active_users',
    'Number of active users'
)

PREDICTION_SUCCESS = Counter(
    'prediction_success_total',
    'Successful predictions'
)

PREDICTION_FAILED = Counter(
    'prediction_failed_total',
    'Failed predictions'
)

NETWORK_IN = Gauge(
    'network_in_mb',
    'Incoming network traffic'
)

NETWORK_OUT = Gauge(
    'network_out_mb',
    'Outgoing network traffic'
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Model accuracy'
)

# START EXPORTER
start_http_server(8000)

print("Prometheus Exporter running on port 8000")

while True:

    try:

        start = time.time()

        response = requests.post(
            "http://localhost:5000/invocations",
            headers={"Content-Type": "application/json"},
            json={
                "inputs": [
                    {
                        "Pclass": 3,
                        "Sex": 1,
                        "Age": 22,
                        "SibSp": 1,
                        "Parch": 0,
                        "Fare": 7.25
                    }
                ]
            }
        )

        latency = time.time() - start

        REQUEST_COUNT.inc()
        PREDICTION_LATENCY.observe(latency)

        if response.status_code == 200:
            PREDICTION_SUCCESS.inc()
        else:
            FAILED_REQUESTS.inc()
            PREDICTION_FAILED.inc()

        # dummy metrics
        CPU_USAGE.set(random.randint(10, 90))
        MEMORY_USAGE.set(random.randint(20, 95))
        ACTIVE_USERS.set(random.randint(1, 100))
        NETWORK_IN.set(random.randint(1, 100))
        NETWORK_OUT.set(random.randint(1, 100))
        MODEL_ACCURACY.set(0.85)

    except:
        FAILED_REQUESTS.inc()

    time.sleep(5)