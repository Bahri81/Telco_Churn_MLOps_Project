from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from pycaret.classification import load_model, predict_model

app = FastAPI(title="Telco Churn MLOps API")

try:
    model = load_model('final_telco_churn_model')
except Exception as e:
    print(f"Model Error: {e}")

class CustomerData(BaseModel):
    Gender: str = Field(..., example="Male")
    Senior_Citizen: int = Field(..., alias="Senior Citizen", example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    Latitude: float = Field(..., example=33.9553)
    Longitude: float = Field(..., example=-118.2573)
    Tenure_Months: int = Field(..., alias="Tenure Months", example=12)
    Contract: str = Field(..., example="Month-to-month")
    Paperless_Billing: str = Field(..., alias="Paperless Billing", example="Yes")
    Payment_Method: str = Field(..., alias="Payment Method", example="Electronic check")
    Monthly_Charges: float = Field(..., alias="Monthly Charges", example=65.5)
    Phone_Service: str = Field(..., alias="Phone Service", example="Yes")
    Multiple_Lines: str = Field(..., alias="Multiple Lines", example="No")
    Internet_Service: str = Field(..., alias="Internet Service", example="Fiber optic")
    Online_Security: str = Field(..., alias="Online Security", example="No")
    Online_Backup: str = Field(..., alias="Online Backup", example="Yes")
    Device_Protection: str = Field(..., alias="Device Protection", example="No")
    Tech_Support: str = Field(..., alias="Tech Support", example="No")
    Streaming_TV: str = Field(..., alias="Streaming TV", example="Yes")
    Streaming_Movies: str = Field(..., alias="Streaming Movies", example="No")

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        input_dict = data.model_dump(by_alias=True)

        for k, v in list(input_dict.items()):
            if isinstance(v, str):
                val_lower = v.lower()
                if val_lower in ['yes', 'male']: 
                    input_dict[k] = 1
                elif val_lower in ['no', 'female']: 
                    input_dict[k] = 0

        tenure = input_dict.get('Tenure Months', 0)
        if tenure <= 12: input_dict['Tenure_Cohort'] = "High_Risk (0-1 year)"
        elif tenure <= 24: input_dict['Tenure_Cohort'] = "Standard (1-2 year)"
        elif tenure <= 48: input_dict['Tenure_Cohort'] = "Loyal (2-4 year)"
        else: input_dict['Tenure_Cohort'] = "Good_Cust (4+ year)"

        expected_cols = list(model.feature_names_in_)
        if 'Churn Value' in expected_cols:
            expected_cols.remove('Churn Value')

        final_data = {}
        for col in expected_cols:

            if col in input_dict and not isinstance(input_dict[col], str):
                final_data[col] = [input_dict[col]]
                
            else:
                is_matched = False
                for key, val in input_dict.items():
                    if isinstance(val, str):
                        if f"{key}_{val}" == col or str(val) == col:
                            final_data[col] = [1.0]
                            is_matched = True
                            break
                
                if not is_matched:
                    final_data[col] = [0.0]

        df_final = pd.DataFrame(final_data)

        predictions = predict_model(model, data=df_final)
        
        pred_value = int(predictions['prediction_label'].iloc[0])
        pred_prob = float(predictions['prediction_score'].iloc[0])
        
        return {
            "status": "success",
            "prediction": "Churn" if pred_value == 1 else "Loyal",
            "risk_probability": round(pred_prob, 4),
            "business_action": "Offer Promotion" if pred_value == 1 else "No action needed"
        }
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"System Error: {str(e)}\n\nExpected Template: {list(model.feature_names_in_)}")