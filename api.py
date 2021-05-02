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


@app.get("/regression/single/")
def single_reg(double: float, speed: float, length: float):
    """
    Make single prediction with query parameters
    ex: localhost:8000/?double=1&speed=2&length=30
    prediction: predicted 0 if no parkinsons, 1 if parkinsons
    predicted_parkinsons: true if closer to 1 (> 0.5)
    """
    inputs = [double, speed, length]
    model = load_regression_model()
    prediction = model.predict([inputs])[0]
    has_parkinsons = 1 if prediction > 0.5 else 0
    return {"prediction": prediction, "predicted_parkinsons": has_parkinsons}


# @app.post("/regression")
# def multi_reg(item: MultipleInputs):
#     inputs = [item.double, item.speed, item.length]
#     model = load_regression_model()
#     predictions = model.predict([inputs])
#     has_parkinsons = True if predictions > 0.5 else False
#     return {"predictions": predictions, "predicted_parkinsons": has_parkinsons}


@app.get("/classification/single/")
def single_class(double: float, speed: float, length: float):
    """
    Make single classification prediction with query parameters
    ex: localhost:8000/?double=1&speed=2&length=30
    prediction: predicted 0 if no parkinsons, 1 if parkinsons
    """
    inputs = [double, speed, length]
    model = load_classifier_model()
    prediction = model.predict([inputs])[0]
    return {"prediction": int(prediction)}


# @app.get("/classification")
# def multi_class(item: MultipleInputs):
#     inputs = [item.double, item.speed, item.length]
#     model = load_classifier_model()
#     predictions = model.predict([inputs])
#     return {"predictions": predictions}


# to run: uvicorn api:app --reload