# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model used for this project was Random Forest classifier. Information about this model can be found in [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). I used the default parameters.
## Intended Use
The idea of the use is to classify based on the employee's information ("workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country") if one employee could have a salary higher or lower than 50K.

## Training Data

The data used for this project is Cesus Boreau public data. This dataset have 32561 registers, a good amount of data to train a model.
## Evaluation Data
The process of data followed a transformation of one hot encoding on columns that weren't numeric and changing the target to 0 and 1 values in order to follow the target classification values. Also a train-test split process giving 80% and 20% respectively.

## Metrics
The metrics are: precision, recall and fbeta. <br>
The results of these metrics are:
* precision: 0.7342814371257484
* recall: 0.6272378516624041
* fbeta: 0.6765517241379311

## Ethical Considerations

As a way to experiment with machine learning techniques, it's acceptable to use sensitive data such as census data. However, it's not ethical to officially use it, for instance, to determine salaries based on certain characteristics. Besides the biases inherent in a machine learning model, it's unethical to rely on a model that could be inaccurate
## Caveats and Recommendations
Data is based on US people so model predictions will not be useful for other countries.