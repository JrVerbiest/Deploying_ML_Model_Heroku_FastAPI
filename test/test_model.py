import sys

from pyrsistent import v
sys.path.insert(0, './src')
sys.path.insert(0, './src/train_model')

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tm_helper import load_pkl
from data import import_data, process_data
from model import train_model, inference, compute_model_metrics


def test_lr_model():
    """test model

    test if the model is a logistic regression model
    """
    PATHMODEL = "./model/lr_model.pkl"

    model = load_pkl(PATHMODEL)
    assert isinstance(model, LogisticRegression)


def test_prediction():
    """test if pred array is a numpy array
    """
    data = import_data("./data/clean_census.csv")
    
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
    pred = inference(model, X_test)

    assert isinstance(pred, (np.ndarray, np.generic))


def test_compute_model_metrics():
    """test if the metrics fall between 0 and 1
    """
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

    PATHCLEANDATA = "./data/clean_census.csv"
    PATHMODEL = "./model/lr_model.pkl"
    PATHENCODER = "./model/lr_encoder.pkl"
    PATHLB = "./model/lr_lb.pkl"

    data = import_data(PATHCLEANDATA)
    model = load_pkl(PATHMODEL)
    encoder = load_pkl(PATHENCODER)
    lb = load_pkl(PATHLB)

    _, test = train_test_split(data, test_size=0.20)

    X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

    pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, pred)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1