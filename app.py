import streamlit as st
import requests
import os

st.set_page_config(
    page_title="Telco Churn Prediction System",
    layout="wide"
)

st.title("Telco Customer Churn Prediction System")
st.markdown("""
This interface connects to the background **FastAPI** microservice to calculate Customer Churn risk in real-time.
Please fill in the customer details below and click the "Predict" button.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("Demographic Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen_text = st.selectbox("Senior Citizen", ["No", "Yes"])
    senior_citizen = 1 if senior_citizen_text == "Yes" else 0
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    
    st.header("Location and Tenure")
    latitude = st.number_input("Latitude", value=33.9553, format="%.4f")
    longitude = st.number_input("Longitude", value=-118.2573, format="%.4f")
    tenure_months = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)

with col2:
    st.header("Service Information")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    st.markdown("### Additional Services")
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

with col3:
    st.header("Billing and Payment")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.5, format="%.2f")

st.markdown("---")

if st.button("Predict Customer Risk", use_container_width=True):
    
    payload = {
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Latitude": latitude,
        "Longitude": longitude,
        "Tenure Months": tenure_months,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies
    }
    
    API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
    
    try:
        with st.spinner("AI Model is calculating, please wait..."):
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                st.subheader("Prediction Results")
                        
                res_col1, res_col2, res_col3 = st.columns(3)
                                
                is_churn = "Churn" in result["prediction"]
                
                with res_col1:
                    st.metric(label="Predicted Status", value=result["prediction"])
                with res_col2:
                    risk_val = float(result["risk_probability"]) * 100
                    risk_pct = f"% {risk_val:.1f}"
                    st.metric(label="Churn Probability (Risk)", value=risk_pct)
                with res_col3:
                    st.metric(label="Recommended Action", value=result["business_action"])
                    
                if is_churn:
                    st.error("ATTENTION: High risk of customer churn! Immediate action required.")
                else:
                    st.success("GREAT: Customer status is stable and loyal.")
                    
            else:
                st.error(f"Server Error! Status Code: {response.status_code}")
                st.write(response.json())
                
    except requests.exceptions.ConnectionError:
        st.error("ERROR: Cannot connect to backend (FastAPI) server! Please ensure \"main.py\" is running with Uvicorn.")