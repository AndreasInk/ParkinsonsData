import pandas as pd
from sklearn import linear_model
import numpy as np
from pathlib import Path
import pyinputplus as pyip

columns = ["double", "speed", "length", "Asym"]

training_dataset = pd.read_csv(Path("./DadDoubleSupport - AiDadMom-2.csv"))

print("Training model...")
regression_model = linear_model.LinearRegression()
X = training_dataset[columns]
Y = training_dataset["sourceName"]
regression_model.fit(X, Y)
print("Model trained.")

std_devs = {a: np.std(training_dataset[a]) for a in ("sourceName", "double")}
print(f"Standard deviations: {std_devs}")

input_list = [pyip.inputNum(f"Input {prompt}: ") for prompt in columns]
pred_price = regression_model.predict([input_list])
has_parkinsons = True if pred_price[0] > 1.5 else False
print(f"Has Parkinsons: {has_parkinsons}")
print("Predicted price:", round(pred_price[0], 2))