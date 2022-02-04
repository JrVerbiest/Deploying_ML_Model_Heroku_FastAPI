
from fastapi.testclient import TestClient
from fastAPI import app

client = TestClient(app)

def test_home():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {"message": "Project 3 - Deploying ML-model Heroku FastAPI."}


def test_predict_one():
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

    response = client.post('/inference', json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": '<=50K'}


def test_predict_two():
    data = {
        "age": 49,
        "workclass": "Local-gov",
        "fnlwgt": 268234,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Protective-serv",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    response = client.post('/inference', json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": '<=50K'}