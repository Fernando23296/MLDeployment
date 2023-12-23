# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
import os
import pickle
import pandas as pd

ldir = os.path.dirname(__file__)

PATH_MODEL = os.path.join(ldir, '../model/rf_model.pkl')
PATH_ENCODER = os.path.join(ldir, '../model/encoder.pkl')
PATH_LB = os.path.join(ldir, '../model/lb.pkl')

# Add code to load in the data.
data = pd.read_csv("../data/new_census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, 
    encoder=encoder, lb=lb
)
# Train and save a model.

model = train_model(X_train, y_train)

y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

print({"precision": precision,
       "recall": recall,
       "fbeta": fbeta})

pickle.dump(model, open(PATH_MODEL, 'wb'))
pickle.dump(encoder, open(PATH_ENCODER, 'wb'))
pickle.dump(lb, open(PATH_LB, 'wb'))
