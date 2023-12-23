import requests

def test_main_route():
    response = requests.get("http://127.0.0.1:8000/")
    assert response.status_code == 200
    assert response.text == '"Welcome"'



def test_prediction_valid_data():
    sample_data = {
        "age": 50,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 83311,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 13,
        "native_country": "United-States"
    }
    response = requests.post("http://127.0.0.1:8000/", json=sample_data)


    assert response.status_code == 200
    assert response.text in ['"Salary less or equal than 50K"', '"Salary higher than 50K"']

def test_prediction_incomplete_data():
    incomplete_data = {
        "age": 50
    }
    response = requests.post("http://127.0.0.1:8000/", json=incomplete_data)
    assert response.status_code == 500
