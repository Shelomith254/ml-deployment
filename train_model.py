import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("data.csv")
X = data[['hours_studied']]
y = data['test_score']

model = LinearRegression()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved")
