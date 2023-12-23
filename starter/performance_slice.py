import os
import json
import joblib
import pandas as pd
from ml.data import process_data
from ml.model import compute_model_metrics
from sklearn.model_selection import train_test_split



PATH_DATA = '../data/new_census.csv'
PATH_MODEL = '../model/'
PATH_METRICS = '../metrics/'

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

def performance_slice(data, model, encoder, lb):

    train, test = train_test_split(data, test_size=0.20)

    with open(os.path.join(PATH_METRICS, 'slice_output.txt'), 'w') as file:
        for feature in cat_features:
            for tf in test[feature].unique():
                df_temp = test[test[feature] == tf]

                X_test, y_test, _, _ = process_data(df_temp,
                                                    cat_features,
                                                    label="salary",
                                                    encoder=encoder,
                                                    lb=lb,
                                                    training=False)

                y_pred = model.predict(X_test)

                precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

                report = {"Feature": feature,
                          "Value": tf,
                          "Precision": precision,
                          "Recall": recall,
                          "Fbeta": fbeta}
                
                file.write(json.dumps(report) + '\n')

if __name__ == '__main__':

    data = pd.read_csv(PATH_DATA)
    model = joblib.load(os.path.join(PATH_MODEL, 'model.pkl'))
    encoder = joblib.load(os.path.join(PATH_MODEL, 'encoder.pkl'))
    lb = joblib.load(os.path.join(PATH_MODEL, 'lb.pkl'))

    performance_slice(data, model, encoder, lb)