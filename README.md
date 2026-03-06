# Customer-Churn-Prediction

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Key Insights](#key-insights)
- [Technical Workflow](#technical-workflow)
- [Model Performance](#model-performance)
- [Interactive Demo](#interactive-demo)
- [Businness Recommendations](#business-recommendations)
- [References](#references)

### Project Overview
This project aims to predict customer churn for a telecommunications company. By analyzing customer demographics, account information and service usage, we identify high risk customers and provide actionable business recommendations to improve retention.

### Data Sources
The dataset used is the WA_Fn-UseC_-Telco-Customer-Churn from Kaggle.
- Size: 7,043 records, 21 predictors
- Target Variables: Churn (Yes/No)

### Key Insights
- Contract Type: "Month-to-month" contracts show the highest churn rates.
- Services: Customers without Online Security or Tech Support are significantly more likely to leave.
- Tenure: New customers (1-12 months) are at the highest risk; churn drops drastically after 5 years of loyalty.
- Charges: High monthly charges correlate with higher churn, specifically when combined with fiber optic internet services.

### Technical Workflow
1. Data Cleaning & Engineering
   - Converted TotalCharges to numeric and handled missing values via median imputation.
   - Grouped tenure into 12-month bins for better categorical analysis.
   - Applied One-Hot Encoding to categorical variables.
2. Handling Class Imbalance
The dataset was imbalanced (approx. 74% No / 26% Yes). I applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the training set, ensuring the model doesn't just "guess" the majority class.
3. Machine Leaning Models
   - Logistic Regression (Baseline)
   - Random Forest
   - XGBoost (Top Performer)

### Model Performance
```Python
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [log_reg_acc, rf_acc, xgb_acc]
})
results
```

### Interactive Demo
The project includes a gradio interface allowing users to input customer data and get real-time churn predictions.
```Python
import gradio as gr
import pandas as pd
import joblib

# 1. POINT TO THE SAVED MODEL (not the csv)
model = joblib.load('churn_model.joblib')
def predict_churn(tenure, monthly, total, contract, security, support, internet):
    # Match the features we used in the training step above
    # Note: We are only using numeric features here to match the simple training above
    df = pd.DataFrame([[tenure, monthly, total]],
                      columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
    prediction = model.predict(df)[0]
    return "⚠️ Likely to Churn" if prediction == 1 else "✅ Likely to Stay"
iface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Tenure (months)", value=12),
        gr.Number(label="Monthly Charges", value=70),
        gr.Number(label="Total Charges", value=1000),
        gr.Dropdown(["Month-to-month","One year","Two year"], label="Contract Type"),
        gr.Dropdown(["Yes","No","No internet service"], label="Online Security"),
        gr.Dropdown(["Yes","No","No internet service"], label="Tech Support"),
        gr.Dropdown(["DSL","Fiber optic","No"], label="Internet Service")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="📊 Telco Customer Churn Prediction"
)
iface.launch(debug=True, share=True)
```

### Business Recommendations 
- Contract Migration: Incentivize "Month-to-month" users to switch to 1 or 2-year contracts through targeted discounts.
- Value-Added Services: Bundle "Online Security" and "Tech Support" for high-paying customers to increase "stickiness."
- Early Intervention: Focus retention marketing on customers within their first 6-12 months of service.
- Service Check-ups: Investigate Fiber Optic service issues, as this segment shows higher-than-average dissatisfaction.

### References
- Dataset Source: Kaggle - Customer-Churn-Prediction
  - [https://www.kaggle.com/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv]
- Tool Documentation: Pandas, Seaborn, Matplotlib, Logistic Regression, Random Forest, XGBoost, Gradio.
   - [https://pandas.pydata.org/docs/]
   - [https://seaborn.pydata.org/]
   - [https://matplotlib.org/]
   - [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html]
   - [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html]
   - [https://xgboost.readthedocs.io/en/release_3.2.0/]
   - [https://www.gradio.app/]
- Analysis Methodology: Inspired by Exploratory Data Analysis (EDA) best practices for content streaming platforms and Machine Learning for prediction.
