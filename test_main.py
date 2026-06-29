from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Telco Churn API operating smoothly"}

def test_predict_churn():

    test_payload = {
        "Gender": "Male",
        "Senior Citizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "Latitude": 33.9553,
        "Longitude": -118.2573,
        "Tenure Months": 12,
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 65.5,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No"
    }
    
    response = client.post("/predict", json=test_payload)
    
    assert response.status_code == 200
    
    data = response.json()
   
    assert "prediction" in data
    assert "risk_probability" in data
    assert "business_action" in data
    
    assert data["prediction"] in ["Churn", "Loyal"]