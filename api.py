from fastapi import FastAPI
import pickle

app = FastAPI()


def load_model():
    model = pickle.load(open("model.sav", "rb"))
    return model


@app.get("/")
def single_prediction(double: float, speed: float, length: float, Asym: float):
    """
    Make single prediction with query parameters
    ex: localhost:8000/?double=1&speed=2&length=30&Asym=0
    """
    inputs = [double, speed, length, Asym]
    model = load_model()
    prediction = model.predict([inputs])[0]
    return {"prediction": prediction}


# to run: uvicorn api:app --reload