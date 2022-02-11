""" FastAPI
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import numpy as np
import pandas as pd

from src.train_model.data import process_data
from src.tm_helper import load_pkl
from src.train_model.model import inference


class Attributes(BaseModel):
    # See Attribute Information on https://archive.ics.uci.edu/ml/datasets/census+income
    age: int
    workclass: Literal["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", 
        "State-gov", "Without-pay", "Never-worked"]
    fnlwgt: int
    education: Literal["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", 
        "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", 
        "Preschool"]
    education_num: int
    marital_status: Literal["Never-married", "Married-civ-spouse", "Divorced","Married-spouse-absent", 
        "Separated", "Married-AF-spouse","Widowed"]
    occupation: Literal["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", 
        "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", 
        "Protective-serv", "Armed-Forces"]
    relationship: Literal["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
    race: Literal["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
    sex: Literal["Female", "Male"]
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Literal["United-States", "Cuba", "Jamaica", "India", "Mexico","Puerto-Rico", "Honduras", 
        "England", "Canada", "Germany", "Iran","Philippines", "Poland", "Columbia", "Cambodia", "Thailand",
        "Ecuador", "Laos", "Taiwan", "Haiti", "Portugal","Dominican-Republic", "El-Salvador", "France", 
        "Guatemala","Italy", "China", "South", "Japan", "Yugoslavia", "Peru","Outlying-US(Guam-USVI-etc)", 
        "Scotland", "Trinadad&Tobago","Greece", "Nicaragua", "Vietnam", "Hong", "Ireland", "Hungary",
        "Holand-Netherlands"]

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Project 3 - Deploying ML-model Heroku FastAPI."}


@app.post('/inference')
def pred(data: Attributes):
    
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

    PATHMODEL = "./model/lr_model.pkl"
    PATHENCODER = "./model/lr_encoder.pkl"
    PATHLB = "./model/lr_lb.pkl"
    
    model = load_pkl(PATHMODEL)
    encoder = load_pkl(PATHENCODER)
    lb = load_pkl(PATHLB)

    columns = ["age","workclass","fnlwgt","education","education_num","marital-status","occupation",
        "relationship","race","sex","capital_gain","capital_loss", "hours-per-week","native-country"]

    data = np.array([[data.age,data.workclass,data.fnlwgt,data.education,data.education_num,
        data.marital_status, data.occupation, data.relationship, data.race, data.sex, data.capital_gain,
        data.capital_loss, data.hours_per_week, data.native_country]])   

    df = pd.DataFrame(data=data, columns=columns)

    X, _, _, _ = process_data(df, categorical_features=cat_features,encoder=encoder, lb=lb, training=False)
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    
    return {"prediction": y}