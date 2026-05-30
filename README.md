# Telco Customer Churn Prediction & Behavioral Analysis

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://customer-churn-prediction-y3diefn7ephwesmffmpn4p.streamlit.app/)

### Table of Contents
- [Project Overview](#project-overview)
- [Data Architecture](#data-architecture)
- [Key Insights and Visualizations](#key-insights-and-visualizations)
- [Technical Stack](#technical-stack)
- [Technical Workflow](#technical-workflow)
- [Model Performance](#model-performance)
- [Business Strategy Recommendations](#business-strategy-recommendations)
- [Setup and How to Run](#setup-and-how-to-run)

---

### Project Overview
This project applies predictive machine learning and exploratory data analysis to identify systemic patterns behind telecom subscriber attrition. By examining the relationship between contract terms, account configurations, and pricing structures, we isolate the primary catalysts for customer churn. The project is finalized with an interactive, production-ready Streamlit frontend deployed to the cloud, allowing stakeholders to evaluate subscriber retention risk in real time.

---

### Data Architecture
The predictive pipeline evaluates the IBM Telco Customer Churn dataset, comprising **7,043 customer records** characterized across 21 structural features:
1. **Demographics:** Gender, Senior Citizen status, Partner, and Dependent relationships.
2. **Account Metrics:** Contract type (Month-to-month, One year, Two year), Tenure (0–72 months), Paperless Billing configurations, and Payment Methods.
3. **Service Profiles:** Internet service architecture (DSL, Fiber Optic, None) and feature add-ons (Online Security, Tech Support, Online Backup, Device Protection, Streaming TV/Movies).
4. **Financial Targets:** `MonthlyCharges` (numeric value), `TotalCharges` (coerced numeric value), and the target objective `Churn` (Yes/No binary indicator).

---

### Key Insights and Visualizations
1. **The Contract Trap:** Contract structure is the strongest structural driver of attrition. Customers on **Month-to-Month contracts** exhibit significantly higher churn rates compared to those on stable One or Two-Year agreements.
2. **Infrastructure Vulnerability:** Subscribers utilizing **Fiber Optic** infrastructure display an elevated churn distribution, pointing to market price sensitivities or service friction points, whereas subscribers with **No Internet Service** maintain the lowest churn footprint.
3. **Support-Driven Retention:** The lack of critical ecosystem features like **Online Security** and **Tech Support** heavily correlates with increased subscriber attrition. Conversely, active engagements in these support structures directly insulate the account from churn.
4. **The Critical First Year:** Tenure distribution plots reveal a heavy concentration of churn within the first **12 months** of account creation. Once a customer crosses the 5-year threshold, retention stability approaches near certainty.

---

### Technical Stack
* **Data Processing & Feature Engineering:** `Pandas`, `NumPy`
* **Exploratory Data Analysis & Visualizations:** `Matplotlib`, `Seaborn`
* **Imbalance Resolution:** `Imbalanced-Learn` (SMOTE)
* **Machine Learning Pipelines:** `Scikit-Learn` (Logistic Regression, Random Forest)
* **Gradient Boosting Frameworks:** `XGBoost`
* **Model Exportation & Storage:** `Joblib`
* **Web UI Framework & Deployment:** `Streamlit`, `Streamlit Community Cloud`

---

### Technical Workflow
1. **Data Cleansing & Cast Optimization:** Structural string abnormalities within the `TotalCharges` matrix were coerced to numeric objects, and resulting null records ($0.15\%$) were filtered out via standard row-wise deletion.
2. **Feature Engineering & Coherence:** Categorical data strings were vectorized into uniform mathematical binaries utilizing **One-Hot Encoding** (`pd.get_dummies`), applying the dummy drop principle to prevent multi-collinearity. 
3. **Class Imbalance Rectification:** To counter the native target imbalance ($74\%$ Active vs. $26\%$ Churned), **SMOTE (Synthetic Minority Over-sampling Technique)** was executed on the training partition to balance class vectors before model training.
4. **Scale Standardisation:** Distance-sensitive models were optimized via `StandardScaler` to align scale variance across continuous inputs (`tenure`, `MonthlyCharges`, `TotalCharges`).
5. **Interactive Interface Development:** Constructed an optimized standalone interface layout in `app.py`, leveraging caching decorator mechanisms (`@st.cache_resource`) to pull saved model weights instantly and serve web-based inferences.

---

### Model Performance
The predictive engine evaluated three algorithmic architectures on stratified test data. Tree-based frameworks outperformed basic generalized linear models, showing exceptional capabilities in processing non-linear service feature boundaries.

| Model Name | Accuracy | Target Profile Summary |
| :--- | :---: | :--- |
| **Random Forest Classifier** | **78.4%** | Highly precise feature weight separation |
| **XGBoost Classifier** | **78.1%** | Robust generalization on continuous feature interactions |
| **Logistic Regression** | **74.9%** | High recall footprint following multi-variable standard scaling |

---

### Business Strategy Recommendations
1. **Contractual Migration Incentives:** Design defensive marketing campaigns focused on transitioning high-risk "Month-to-Month" subscribers into structured 1-Year agreements using targeted promotional discounts.
2. **Strategic Ecosystem Bundling:** Package **Online Security** and **Tech Support** extensions natively into high-tier packages, as these add-ons dramatically minimize the probability of account abandonment.
3. **First-Year Retention Focus:** Deploy automated customer success workflows and check-ins tailored specifically for new accounts within their initial 12-month lifecycle window to mitigate early-stage churn.

---

### Setup and How to Run

#### 1. Environment Setup
Clone this repository to your local directory and install the pinned dependencies:
```bash
git clone [https://github.com/tanusattri/Customer-Churn-Prediction.git](https://github.com/tanusattri/Customer-Churn-Prediction.git)
cd Customer-Churn-Prediction
pip install -r requirements.txt
