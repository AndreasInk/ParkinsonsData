import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
from pathlib import Path
import pyinputplus as pyip

columns = ["double", "speed", "length", "Asym"]

df = pd.read_csv(Path("./DadDoubleSupport - AiDadMom-2.csv"))
# shuffle rows
df = df.sample(frac=1).reset_index(drop=True)

# split the data into train and test
ratio = 0.9
split_row = round(ratio * df.shape[0])
train_df = df.iloc[:split_row]
test_df = df.iloc[split_row:]

print("Training model...")
regression_model = linear_model.LinearRegression()
X = train_df[columns]
Y = train_df["sourceName"]
regression_model.fit(X, Y)
print("Model trained.")

std_devs = {a: np.std(train_df[a]) for a in ("sourceName", "double")}
print(f"Standard deviations: {std_devs}")

# see how well the model did
pred_y = regression_model.predict(test_df[columns])
real_y = test_df["sourceName"]
assert len(pred_y) == len(real_y), "Must have same number of predictions"
print(f"performance (r2) score: {r2_score(real_y, pred_y)}")

if False:
    input_list = [pyip.inputNum(f"Input {prompt}: ") for prompt in columns]
    pred_price = regression_model.predict([input_list])
    has_parkinsons = True if pred_price[0] > 1.5 else False
    print(f"Has Parkinsons: {has_parkinsons}")
    print("Predicted price:", round(pred_price[0], 2))