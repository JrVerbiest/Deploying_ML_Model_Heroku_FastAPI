""" sample request
"""

import requests


data = {
    "age": 59,
    "workclass": "Private",
    "fnlwgt": 109015,
    "education": "HS-grad",
    "education_num": 9,
    "marital_status": "Divorced",
    "occupation": "Tech-support",
    "relationship": "Unmarried",
    "race": "White",
    "sex": "Female",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
    }


response = requests.post('http://127.0.0.1:8000/inference', json=data)
assert response.status_code == 200

print(f"Response code: {response.status_code}")
print(f"Response body: {response.json()}")