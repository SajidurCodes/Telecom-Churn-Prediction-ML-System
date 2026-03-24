from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json

from src.features import create_features

app = FastAPI(title="Telecom Churn Prediction API")


# Load model and metadata
model = joblib.load("./Models/lightgbm_churn_model.pkl")  

with open("./Models/metadata.json", "r") as f:
    metadata = json.load(f)

feature_names = metadata["features"]


class CustomerData(BaseModel):
    AccountWeeks: int
    ContractRenewal: int
    DataPlan: int
    DataUsage: float
    CustServCalls: int
    DayMins: float
    DayCalls: int
    MonthlyCharge: float
    OverageFee: float
    RoamMins: float


@app.get("/")
def home():
    return {"message": "Telecom Churn Prediction API is running."}


@app.post("/predict")
def predict(customer: CustomerData):
    input_dict = customer.dict()
    df = pd.DataFrame([input_dict])

    df = create_features(df)
    X = df[feature_names]

    churn_probability = float(model.predict(X)[0])
    churn_prediction = int(churn_probability >= 0.5)

    return {
        "churn_probability": round(churn_probability, 4),
        "churn_prediction": churn_prediction
    }