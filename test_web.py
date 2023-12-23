import requests

def test_main_route():
    response = requests.get("https://proy-final-4.onrender.com/")
    assert response.status_code == 200
    assert response.text == '"Welcome"'



def test_prediction_valid_data_higher_than_50K():
  
    sample_data = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    }
    response = requests.post("https://proy-final-4.onrender.com/", json=sample_data)


    assert response.status_code == 200
    assert response.text == '"Salary higher than 50K"'


def test_prediction_salary_less_than_50K():
    sample_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = requests.post("https://proy-final-4.onrender.com/", json=sample_data)
    assert response.status_code == 200
    assert response.text == '"Salary less or equal than 50K"'

def test_prediction_incomplete_data():
    incomplete_data = {
        "age": 50
    }
    response = requests.post("https://proy-final-4.onrender.com/", json=incomplete_data)
    assert response.status_code == 500
