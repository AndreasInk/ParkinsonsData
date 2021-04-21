import pandas as pd
from sklearn import linear_model
import numpy as np

training_dataset = pd.read_csv("downloads/All5.csv")

print ("Training model...")
regression_model = linear_model.LinearRegression() 
X = training_dataset[['double', 'speed','length', 'Asym']]
Y = training_dataset['sourceName']
regression_model.fit(X, Y) 
a = np.array([])
for key, value in training_dataset:

    a = a + (i['sourceName'])
    a = a + (i['double'])
print(np.std(a))
print ("Model trained.")

input_list = []
for x in ("first", "second", "third", "fourth"):
    input_area = float(input(f"Enter {x}: "))
    input_list.append(input_area)
proped_price = regression_model.predict([input_list])
print ("Proped price:", round(proped_price[0], 2))