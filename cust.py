#import the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
#Streamlined large-scale data pipelines using SQL and Python, reducing processing time by [X] percent while ensuring 100 percent accuracy for reporting.

#load the first_telc
import pandas as pd
telco_base_data = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Customer Churn Prediction\WA_Fn-UseC_-Telco-Customer-Churn.csv')

#Look at the top 5 records of data
telco_base_data.head()

#Checking the attribute: shape
telco_base_data.shape

#Checking the attribute: values
telco_base_data.columns.values

# Checking the attribute: data types of all the columns
telco_base_data.dtypes

# Check the descriptive statistics of numeric variables
telco_base_data.describe()
#Observation: SeniorCitizen is actually a categorical hence the 25%-50%-75% distribution is not proper.
#75% customers have tenure less than 55 months
#Average Monthly charges are USD 64.76 whereas 25% customers pay more than USD 89.85 per month

# Observation: The plot shows that the dataset is imbalanced, with significantly more customers who did not churn compared to those who did.
telco_base_data['Churn'].value_counts().plot(kind='barh', figsize=(8, 6),color=['#00bfae', '#ff6f61'])
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variable", labelpad=14)
plt.title("Count of TARGET Variable per category", y=1.02);

# Observation: Around 26% of customers have churned, while about 74% have stayed with the company, indicating class imbalance in the dataset.
100*telco_base_data['Churn'].value_counts()/len(telco_base_data['Churn'])

#calculates and displays the count for each category (Yes/No) within the 'Churn' column
telco_base_data['Churn'].value_counts()

# Concise Summary of the dataframe, as we have too many columns, we are using the verbose = True mode
telco_base_data.info(verbose = True)

#observation: Missing value percentages are calculated for all columns and displayed, quickly identifying the features that require imputation or removal, with a visible spike at the TotalCharges column.
missing = pd.DataFrame((telco_base_data.isnull().sum())*100/telco_base_data.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot(x='index',y=0,data=missing, color='#00bfae')
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.xlabel("Features")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

#Data Cleaning
#Step-1: Create a copy of base data for manupulation & processing
telco_data = telco_base_data.copy()

#Step-2: Total Charges should be numeric amount. Let's convert it to numerical data type
telco_data.TotalCharges = pd.to_numeric(telco_data.TotalCharges, errors='coerce')
telco_data.isnull().sum()

#Step-3: As we can see there are 11 missing values in TotalCharges column. Let's check these records
telco_data.loc[telco_data ['TotalCharges'].isnull() == True]

#Step-4: Missing Value Treatement
#Removing missing values
telco_data.dropna(how = 'any', inplace = True)

#Step-5: Divide and Get the max tenure
print(telco_data['tenure'].max())

# Group the tenure in bins of 12 months
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
telco_data['tenure_group'] = pd.cut(telco_data.tenure, range(1, 80, 12), right=False, labels=labels)
telco_data['tenure_group'].value_counts()

#Step-6: Remove columns not required for processing
#drop column customerID and tenure
# FIX: Removed axis=1 parameter to fix the ValueError conflict with the columns argument
telco_data.drop(columns=['customerID', 'tenure'], inplace=True)
telco_data.head()

#Data Exploration
#1. Plot distibution of individual predictors by churn
#Univariate Analysis
custom_palette={'No': '#00bfae', 'Yes': '#ff6f61'}
for i, predictor in enumerate(telco_data.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i,figsize=(10, 6))
    ax = sns.countplot(data=telco_data, x=predictor, hue='Churn', palette=custom_palette)
    if len(telco_data[predictor].unique()) > 5:
        plt.xticks(rotation=45, ha='right')
    plt.title(f"Churn Distribution by {predictor}")
    plt.show()

#Step-2: Convert the target variable 'Churn' in a binary numeric variable i.e. Yes=1 ; No = 0
telco_data['Churn'] = np.where(telco_data.Churn == 'Yes',1,0)
telco_data.head()

#Step-3: Convert all the categorical variables into dummy variables
telco_data_dummies = pd.get_dummies(telco_data)
telco_data_dummies.head()

#Relationship between Monthly Charges and Total Charges
custom_palette = {
    0: '#00bfae',  # Non-Churn color
    1: '#ff6f61'  # Churn color
}
sns.lmplot(
    data=telco_data_dummies,
    x='MonthlyCharges',
    y='TotalCharges',
    hue='Churn',
    palette=custom_palette,
    fit_reg=False,
    height=6,
    aspect=1.5
)
plt.title("Total Charges vs. Monthly Charges by Churn Status")
plt.show()

#Churn by Monthly Charges and Total Charges
Mth = sns.kdeplot(telco_data_dummies.MonthlyCharges[(telco_data_dummies["Churn"] == 0)], color="#00bfae", fill = True)
Mth = sns.kdeplot(telco_data_dummies.MonthlyCharges[(telco_data_dummies["Churn"] == 1)],ax=Mth, color="#ff6f61", fill=True)
Mth.legend(["No Churn","Churn"], loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')
plt.show()
#Insight: Churn is high when Monthly Charges ar high

Tot = sns.kdeplot(telco_data_dummies.TotalCharges[(telco_data_dummies["Churn"] == 0) ], color="#00bfae", fill = True)
Tot = sns.kdeplot(telco_data_dummies.TotalCharges[(telco_data_dummies["Churn"] == 1) ], ax =Tot, color="#ff6f61", fill= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total charges by churn')
plt.show()
#Insight: Higher Monthly Charge at lower tenure results into lower Total Charge. Hence, all these 3 factors viz Higher Monthly Charge, Lower tenure and Lower Total Charge are linkd to High Churn.

#Build a corelation of all predictors with 'Churn'
corr_values = telco_data_dummies.corr()['Churn'].sort_values(ascending=False)
colors = ['#ff6f61' if x > 0 else '#00bfae' for x in corr_values]
plt.figure(figsize=(20, 8))
corr_values.plot(kind='bar', color=colors)
plt.title("Correlation of Features with Churn", fontsize=16)
plt.ylabel("Correlation Coefficient", labelpad=10)
plt.xlabel("Features", labelpad=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
#Derived-Insight: HIGH Churn seen in case of Month to month contracts, No online security, No Tech support, First year of subscription and Fibre Optics Internet.
#Low Churn is seens in case of Long term contracts, Subscriptions without internet service and The customers engaged for 5+ years
#Factors like Gender, Availability of PhoneService and of multiple lines have alomost NO impact on Churn

#Above insights are also evident from the Heatmap below
plt.figure(figsize=(12,12))
sns.heatmap(telco_data_dummies.corr(), cmap="coolwarm", annot=False, fmt=".2f")
plt.title("Correlation Matrix of Features", fontsize=16)
plt.show()

#Load & Clean Data
import pandas as pd

# Load dataset
# FIX: Updated server /content path to the local project path
telco = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Customer Churn Prediction\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle TotalCharges (common issue: blank values)
telco['TotalCharges'] = pd.to_numeric(telco['TotalCharges'], errors='coerce')
telco['TotalCharges'] = telco['TotalCharges'].fillna(telco['TotalCharges'].median())

# Convert target to numeric
telco['Churn'] = telco['Churn'].map({'Yes': 1, 'No': 0})

# Drop unnecessary ID
telco = telco.drop(['customerID'], axis=1)

# Convert categoricals → one-hot encoding
telco_dummies = pd.get_dummies(telco, drop_first=True)

print("Shape:", telco_dummies.shape)
telco_dummies.head()


#Train/Test Split
from sklearn.model_selection import train_test_split

X = telco_dummies.drop("Churn", axis=1)
y = telco_dummies["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


#Train ML Models (LR, RF, XGB)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE # Import SMOTE

# Apply SMOTE to Fix Imbalance
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Logistic Regression (scale required)
# Note: X_train and X_test had numerical columns scaled in a previous step (cell VUB_znVqz-9k).
# X_train_res now contains resampled data with scaled numerical and unscaled categorical features.
# This step applies another scaler, effectively scaling all features (numerical and categorical) for Logistic Regression.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=300)
log_reg.fit(X_train_scaled, y_train_res)
y_pred_lr = log_reg.predict(X_test_scaled)
log_reg_acc = accuracy_score(y_test, y_pred_lr)

# Random Forest
# Random Forest does not require features to be scaled. X_train_res contains resampled data.
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)

# XGBoost
# XGBoost is also a tree-based model and generally does not require feature scaling.
xgb = XGBClassifier(eval_metric='logloss')
xgb.fit(X_train_res, y_train_res)
y_pred_xgb = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, y_pred_xgb)

print("LR Accuracy:", log_reg_acc)
print("RF Accuracy:", rf_acc)
print("XGBoost Accuracy:", xgb_acc)

#Comparison Table
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [log_reg_acc, rf_acc, xgb_acc]
})

results


#Classification Report
print("Classification Report for Random Forest")
print(classification_report(y_test, y_pred_rf))


print("----- FINAL PROJECT CONCLUSION -----\n")

print("📌 Key Insights From Data:")
print("- Month-to-month contract customers churn the most.")
print("- Lack of Online Security & Tech Support increases churn.")
print("- Fibre Optic users show higher churn.")
print("- Customers with long tenure (5+ years) rarely churn.\n")

print("📌 Machine Learning Outcome:")
print(f"- Logistic Regression Accuracy: {log_reg_acc:.4f}")
print(f"- Random Forest Accuracy: {rf_acc:.4f}")
print(f"- XGBoost Accuracy: {xgb_acc:.4f}")
print("- Random Forest / XGBoost performed the best.\n")

print("📌 Business Interpretation:")
print("- Encourage customers to move from monthly to yearly contracts.")
print("- Offer discounts on security & support add-ons.")
print("- Improve Fibre Optic service experience.")
print("- Focus retention on new customers in their first year.\n")

print("🎯 Final Takeaway:")
print("A smart churn prediction model helps telecom companies identify at-risk customers early and take necessary retention actions.\n")



#Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Random Forest - Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()


#Feature Importance Plot (from RF)
import seaborn as sns
import matplotlib.pyplot as plt

feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(data=feat_imp.head(15), x="Importance", y="Feature")
plt.title("Top 15 Most Important Features for Churn Prediction")
plt.show()


import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load the dataset
# FIX: Updated server path to local path
telco_base_data = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Customer Churn Prediction\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2️⃣ Fix TotalCharges (convert to numeric) & drop customerID
telco_base_data['TotalCharges'] = pd.to_numeric(telco_base_data['TotalCharges'], errors='coerce')
telco_base_data['TotalCharges'] = telco_base_data['TotalCharges'].fillna(telco_base_data['TotalCharges'].median())
telco_base_data = telco_base_data.drop('customerID', axis=1)

# 3️⃣ Convert target variable to numeric
telco_base_data['Churn'] = telco_base_data['Churn'].map({'Yes': 1, 'No': 0})

# 4️⃣ Create dummy variables (drop first to avoid multicollinearity)
telco_data_dummies = pd.get_dummies(telco_base_data, drop_first=True)

# 5️⃣ Optional: check shape
print(telco_data_dummies.shape)
telco_data_dummies.head()


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Example: using your cleaned dummy data
X = telco_data_dummies.drop("Churn", axis=1)
y = telco_data_dummies["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric columns
scaler = StandardScaler()
num_cols = ["tenure","MonthlyCharges","TotalCharges"]
X_train = X_train.copy() # Avoid SettingWithCopyWarning
X_test = X_test.copy()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Train models
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(eval_metric="logloss")
xgb_model.fit(X_train, y_train)


#Apply SMOTE to Fix Imbalance
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("\nAfter SMOTE:", y_train_res.value_counts())


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your data
# FIX: Updated server path to local path
df_train = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Customer Churn Prediction\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Simple preprocessing for the demo (Selecting only numeric columns for now)
# In a real project, you'd want to encode the text columns too!
features = ['tenure', 'MonthlyCharges', 'TotalCharges']
df_train['TotalCharges'] = pd.to_numeric(df_train['TotalCharges'], errors='coerce').fillna(0)

X = df_train[features]
y = df_train['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# SAVE THE MODEL properly
joblib.dump(model, 'churn_model.joblib')
print("✅ Model trained and saved as 'churn_model.joblib'")

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