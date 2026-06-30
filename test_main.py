from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

BASE_PAYLOAD = {
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
    "Streaming Movies": "No",
}


def _risk(payload):
    """POST a payload and return the risk_probability (asserts a 200 first)."""
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()["risk_probability"]


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"mesaj": "Telco Churn API operating smoothly"}


def test_predict_response_shape():
    resp = client.post("/predict", json=BASE_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "risk_probability" in data
    assert "business_action" in data
    assert data["prediction"] in ["Churn", "Loyal"]
    assert 0.0 <= data["risk_probability"] <= 1.0

def test_internet_service_affects_prediction():
    fiber = dict(BASE_PAYLOAD, **{"Internet Service": "Fiber optic"})
    dsl = dict(BASE_PAYLOAD, **{"Internet Service": "DSL"})
    assert _risk(fiber) != _risk(dsl), \
        "Internet Service has no effect -> it is being dropped before the model."


def test_payment_method_affects_prediction():
    echeck = dict(BASE_PAYLOAD, **{"Payment Method": "Electronic check"})
    card = dict(BASE_PAYLOAD, **{"Payment Method": "Credit card (automatic)"})
    assert _risk(echeck) != _risk(card), \
        "Payment Method has no effect -> it is being dropped before the model."


def test_contract_affects_prediction():
    m2m = dict(BASE_PAYLOAD, **{"Contract": "Month-to-month"})
    two_year = dict(BASE_PAYLOAD, **{"Contract": "Two year"})
    assert _risk(m2m) != _risk(two_year), \
        "Contract has no effect -> the #1 churn driver is being dropped."


def test_tenure_affects_prediction():
    new = dict(BASE_PAYLOAD, **{"Tenure Months": 2})
    loyal = dict(BASE_PAYLOAD, **{"Tenure Months": 60})
    assert _risk(new) != _risk(loyal), \
        "Tenure has no effect -> tenure / Tenure_Cohort is being dropped."
