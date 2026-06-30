from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import pandas as pd
from pycaret.classification import load_model, predict_model

app = FastAPI(title="Telco Churn MLOps API")

model = load_model('final_telco_churn_model')

BINARY_COLS = [
    "Partner", "Dependents", "Phone Service", "Multiple Lines",
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies", "Paperless Billing",
]


def create_tenure_cohort(tenure):
    if tenure <= 12:
        return "High_Risk (0-1 year)"
    elif tenure <= 24:
        return "Standard (1-2 year)"
    elif tenure <= 48:
        return "Loyal (2-4 year)"
    else:
        return "Good_Cust (4+ year)"


@app.get("/")
def read_root():
    return {"mesaj": "Telco Churn API operating smoothly"}


@app.post("/predict")
def predict_churn(data: Dict[str, Any]):
    try:
        df = pd.DataFrame([data])

        if "Tenure Months" in df.columns:
            df["Tenure_Cohort"] = df["Tenure Months"].apply(create_tenure_cohort)

        for col in BINARY_COLS:
            if col in df.columns:
                df[col] = df[col].replace({"Yes": 1, "No": 0})

        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].replace({"Male": 1, "Female": 0})

        if "Contract" in df.columns:
            df["Contract"] = df["Contract"].replace(
                {"Month-to-month": 0, "One year": 1, "Two year": 2}
            )

        predictions = predict_model(model, data=df)

        pred_label = int(predictions["prediction_label"].iloc[0])
        pred_score = float(predictions["prediction_score"].iloc[0])

        return {
            "prediction": "Churn" if pred_label == 1 else "Loyal",
            "risk_probability": pred_score,
            "business_action": "Retention Campaign" if pred_label == 1 else "No Action Needed",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
