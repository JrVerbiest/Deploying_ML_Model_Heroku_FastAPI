""" Script to train machine learning model.

"""

import sys
sys.path.insert(0, "./src/")
sys.path.insert(1, "./train_model")

print(sys.path)

from sklearn.model_selection import train_test_split

from data import import_data, process_data
from model import train_model, inference, compute_model_metrics
from tm_helper import save_pkl

# parameters
PATHCLEANDATA = "../data/clean_census.csv"
PATHMODEL = "../model/lr_model.pkl"

# load in the data.
data = import_data(PATHCLEANDATA)

# split data
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

X_train, y_train, encoder_train, lb_train = process_data(train, categorical_features=cat_features, label="salary", training=True)
X_test, y_test, encoder_test, lb_test = process_data(test, categorical_features=cat_features, label="salary", training=False, encoder=encoder_train, lb=lb_train)

model = train_model(X_train, y_train)

save_pkl(model, PATHMODEL)

pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, pred)

print("Metrics: precision = %s,  recall = %s and fbeta = %s" %(round(precision,2), round(recall,2), round(fbeta,2)))
