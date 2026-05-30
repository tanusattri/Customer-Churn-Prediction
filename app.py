import streamlit as st
import pandas as pd
import joblib

# 1. Set Page Configuration
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# 2. Load the Saved Model
@st.cache_resource  # Caches the model so it doesn't reload on every interaction
def load_churn_model():
    # Make sure 'churn_model.joblib' is in the same folder as this script
    return joblib.load("churn_model.joblib")

try:
    model = load_churn_model()
except Exception as e:
    st.error(f"Error loading model file: {e}")

# 3. Application Interface Header
st.title("📊 Telco Customer Churn Prediction App")
st.markdown("Provide the customer metrics below to evaluate their risk of churning.")
st.write("---")

# 4. User Input Form Layout
st.subheader("Customer Metrics")

# Splitting inputs into two columns for a neat UI layout
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12, help="Number of months the customer has stayed with the company")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)

with col2:
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0)
    
    # Optional styling placeholders for unmatched visual categories from the script's basic training block
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

st.write("---")

# 5. Prediction Execution
if st.button("Predict Churn Status", type="primary"):
    try:
        # Construct the exact feature structure matching the 3 features used in your model training step
        input_data = pd.DataFrame(
            [[tenure, monthly_charges, total_charges]], 
            columns=['tenure', 'MonthlyCharges', 'TotalCharges']
        )
        
        # Run prediction
        prediction = model.predict(input_data)[0]
        
        st.subheader("Analysis Outcome")
        if prediction == 1:
            st.error("🚨 **High Risk Warning:** This customer is highly likely to churn!")
        else:
            st.success("✅ **Stable Customer Status:** This customer is likely to stay.")
            
    except NameError:
        st.warning("Model pipeline is unavailable. Verify that your 'churn_model.joblib' file exists in this directory.")