import sklearn
import pandas as pd
import joblib

with open("artifacts/lead_model_lr.pkl", "rb") as f:
    model = joblib.load(f)

X = pd.read_csv("artifacts/X_test.csv")
y = pd.read_csv("artifacts/y_test.csv")
print(model.predict(X.head(5)), y.head(5))
