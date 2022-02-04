""" Test performance on differen slices

"""

import sys
sys.path.insert(0, "./src/")
sys.path.insert(1, "./train_model")

from sklearn.model_selection import train_test_split

from data import import_data, process_data
from model import train_model, inference, compute_model_metrics
from tm_helper import load_pkl

# parameters
PATHCLEANDATA = "../data/clean_census.csv"
PATHMODEL = "../model/lr_model.pkl"
PATHENCODER = "../model/lr_encoder.pkl"
PATHLB = "../model/lr_lb.pkl"
PATHSLICEOUTPUT = "../slice_output.txt"


def run():

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

    # load in the data.
    data = import_data(PATHCLEANDATA)
    
    model = load_pkl(PATHMODEL)
    encoder = load_pkl(PATHENCODER)
    lb = load_pkl(PATHLB)

    # split data
    _, test = train_test_split(data, test_size=0.20)


    slice_score = []
    for feature in cat_features:
        for element in test[feature].unique():
            df = test[test[feature] == element]

            X_test, y_test, _, _ = process_data(df,categorical_features=cat_features,label="salary", encoder=encoder, lb=lb, training=False)
            pred = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, pred)

            slice_score.append("Feature: {} - item: {} - precision:{} recall:{} fbeta:{}".format(feature, element, round(precision,2), round(recall,2), round(fbeta,2)))
    

    with open(PATHSLICEOUTPUT, 'w') as slice_output_file:
        for score in slice_score:
            slice_output_file.write(score + '\n')
        slice_output_file.close()


if __name__ == '__main__':
    run()