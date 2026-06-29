from fastapi import FastAPI
from typing import Dict, Any
import pandas as pd
from pycaret.classification import load_model, predict_model

app = FastAPI(title="Telco Churn MLOps API")

# Modeli yükle
model = load_model('final_telco_churn_model')

def create_tenure_cohort(tenure):
    if tenure <= 12: return "High_Risk (0-1 year)"
    elif tenure <= 24: return "Standard (1-2 year)"
    elif tenure <= 48: return "Loyal (2-4 year)"
    else: return "Good_Cust (4+ year)"

@app.get("/")
def read_root():
    return {"mesaj": "Telco Churn API operating smoothly"}

@app.post("/predict")
def predict_churn(data: Dict[str, Any]):
   
    df_input = pd.DataFrame([data])
    
    if 'Tenure Months' in df_input.columns:
        df_input['Tenure_Cohort'] = df_input['Tenure Months'].apply(create_tenure_cohort)
    
    predictions = predict_model(model, data=df_input)
    
    pred_label = predictions['prediction_label'].iloc[0]
    pred_score = predictions['prediction_score'].iloc[0]
    
    return {
        "prediction": "Churn" if pred_label == 1 else "Loyal",
        "risk_probability": float(pred_score),
        "business_action": "Retention Campaign" if pred_label == 1 else "No Action Needed"
    }