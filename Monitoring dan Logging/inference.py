import requests
import time

url = "http://127.0.0.1:5000/invocations"

data = {
    "dataframe_records": [
        {
            "Pclass": 3,
            "Sex": 1,
            "Age": 22,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 7.25,
            "Embarked": 0
        }
    ]
}

start = time.time()

response = requests.post(
    url,
    json=data,
    headers={"Content-Type": "application/json"}
)

end = time.time()

print("Response:", response.json())
print("Latency:", end - start)