import pandas as pd
import pytest
from starter.ml.data import process_data
from starter.ml.model import inference
import os
import pickle
from sklearn.model_selection import train_test_split

MODEL_PATH = 'model/'

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture()
def data():
    return pd.read_csv("data/new_census.csv")


def test_registers(data):
    rows = data.shape[0]
    columns = data.shape[1]
    try:
        assert rows>0
        assert columns>0
        print('SUCCESS: There are registers in dataframe')

    except:
        print('ERROR: No registers in dataframe')

def test_model_existance():
    
        expected_files = ['encoder.pkl', 'model.pkl', 'lb.pkl']

        for file_name in expected_files:
            file_path = os.path.join(MODEL_PATH, file_name)
            try:
                with open(file_path, 'rb') as file:
                    f = pickle.load(file)

                print(f'SUCCESS: {file_path} exists')
            except FileNotFoundError as e:
                print(f"ERROR: {file_path} doesn't exist")
                raise e


def test_model(data):
    train, test = train_test_split(data, test_size=0.20)

    with open('model/model.pkl', 'rb') as m:
        model = pickle.load(m)
    
    with open('model/encoder.pkl', 'rb') as e:
        encoder = pickle.load(e)

    with open('model/lb.pkl', 'rb') as l:
        lb = pickle.load(l)

    X_test, y_test, encoder, lb = process_data(test, 
                                               categorical_features=cat_features, 
                                               label="salary", 
                                               training=False,
                                               encoder=encoder, 
                                               lb=lb)
    y_pred = inference(model, X_test)
    
    try:
        assert(y_test.shape == y_pred.shape)
        print("SUCCESS: Model inference was done succesfully")

    except:
        print("ERROR: There was an error in model prediction")

