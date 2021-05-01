from fastapi import FastAPI
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = FastAPI()


def load_model():
    model = pickle.load(open("model.sav", "rb"))
    return model


@app.get("/")
def single_prediction(double: float, speed: float, length: float, Asym: float):
    """
    Make single prediction with query parameters
    ex: localhost:8000/?double=1&speed=2&length=30&Asym=0
    prediction: predicted 1 if no parkinsons, 2 if parkinsons
    predicted_parkinsons: true if closer to 2 (> 1.5)
    """
    inputs = [double, speed, length, Asym]
    model = load_model()
    prediction = model.predict([inputs])[0]
    has_parkinsons = True if prediction > 1.5 else False
    return {"prediction": prediction, "predicted_parkinsons": has_parkinsons}


# to run: uvicorn api:app --reload