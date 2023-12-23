# Put the code for your API here.
import os
import pickle
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()

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

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model/')

class Sample(BaseModel):
    age: int = Field(None, example=50, description="The age of the individual")
    workclass: str = Field(None, example='Self-emp-not-inc', description="The work class")
    fnlgt: int = Field(None, example=83311, description="Final weight")
    education: str = Field(None, example='Bachelors', description="The highest level of education achieved")
    education_num: int = Field(None, example=13, description="The number of educational years completed")
    marital_status: str = Field(None, example='Married-civ-spouse', description="Marital status")
    occupation: str = Field(None, example='Exec-managerial', description="The occupation")
    relationship: str = Field(None, example='Husband', description="Relationship status")
    race: str = Field(None, example='White', description="Race")
    sex: str = Field(None, example='Male', description="Sex")
    capital_gain: int = Field(None, example=0, description="Capital gains recorded")
    capital_loss: int = Field(None, example=0, description="Capital losses recorded")
    hours_per_week: int = Field(None, example=13, description="Number of hours worked per week")
    native_country: str = Field(None, example='United-States', description="Native country")


@app.get("/")
async def principal():
  return "Welcome"


@app.post("/")
async def predict(sample: Sample):
    
    new_sample = {}
    for k,v in sample:
        print(k.replace('_','-'),v)
        new_sample[k.replace('_', '-')] = [v]

    data = pd.DataFrame.from_dict(new_sample)

    model = pickle.load(open(os.path.join(MODEL_PATH, 'model.pkl'), 'rb'))
    encoder = pickle.load(open(os.path.join(MODEL_PATH, 'encoder.pkl'), 'rb'))
    lb = pickle.load(open(os.path.join(MODEL_PATH, 'lb.pkl'), 'rb'))

    X, _, _, _ = process_data(data, 
                              categorical_features=cat_features, 
                              label=None, 
                              training=False, 
                              encoder=encoder, 
                              lb=lb)

    pred = inference(model, X)

    if pred[0] == 0:
        return 'Salary less or equal than 50K'
    else:
        return 'Salary higher than 50K'

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)