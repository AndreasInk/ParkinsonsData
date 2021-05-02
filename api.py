from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path

app = FastAPI()


class MultipleInputs(BaseModel):
    double: List[float]
    speed: List[float]
    length: List[float]


def load_regression_model():
    import_dir = Path("models/reg_model.sav")
    model = pickle.load(open(import_dir, "rb"))
    return model


def load_classifier_model():
    import_dir = Path("models/class_model.sav")
    model = pickle.load(open(import_dir, "rb"))
    return model


@app.get("/single/")
def single_pred(double: float, speed: float, length: float):
    """
    Make single prediction with query parameters
    ex: localhost:8000/?double=1&speed=2&length=30
    predicted 0 if no parkinsons, 1 if parkinsons
    """
    inputs = [double, speed, length]
    reg_model = load_regression_model()
    class_model = load_classifier_model()

    reg_pred = reg_model.predict([inputs])[0]
    class_pred = class_model.predict([inputs])[0]
    return {
        "regression_prediction": float(reg_pred),
        "classification_prediction": int(class_pred),
    }


@app.post("/")
def multi_pred(item: MultipleInputs):
    # reshape inputs
    model_input = []
    for d, s, l in zip(item.double, item.speed, item.length):
        model_input.append([d, s, l])
    reg_model = load_regression_model()
    class_model = load_classifier_model()

    reg_pred = reg_model.predict(model_input)
    class_pred = class_model.predict(model_input)
    return {
        "regression_predictions": [float(i) for i in list(reg_pred)],
        "classification_predictions": [int(i) for i in list(class_pred)],
    }
