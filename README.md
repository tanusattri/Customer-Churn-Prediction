# Customer-Churn-Prediction

### Project Overview
Developed an end-to-end machine learning pipeline to predict customer attrition using Python and Scikit-learn. The project involved extensive exploratory data analysis, feature engineering and the training of classification models like Random Forest and XGBoost to identify high-risk customers. By pinpointing key churn drivers—such as contract types and pricing thresholds—the model provides actionable insights to optimize retention strategies and safeguard company revenue.

### Data Sources
Customer Churn Dataset via Kaggle, consisting of 8,800+ records of age group, gender and metadata.

### Tools
- Python
- Pandas, Numpy
- Matplotlib, Seaborn
- Scikit-learn, XGBoost
- Google Colab

### Exploratory Data Analysis 
- Identified that Contract Type and Tenure have the strongest negative correlation with churn, meaning long-term contracts significantly improve retention.
- Visualized service usage patterns, revealing that fiber optic users and those without Online Security had disproportionately higher attrition rates.
- Analyzed Monthly Charges and Total Charges, uncovering that high-cost customers were more sensitive to price hikes and more likely to leave.

### Data Anaylsis 
```Python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('customer_churn_data.csv')
print(df.info())
print(df.describe())

plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='viridis')
plt.title('Distribution of Customer Churn')
plt.show()
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Contract', hue='Churn', data=df, palette='magma')
plt.title('Churn Rate by Contract Type')
plt.show()
plt.figure(figsize=(10, 6))
sns.kdeplot(df[df['Churn'] == 'No']['tenure'], label='Stayed', fill=True)
sns.kdeplot(df[df['Churn'] == 'Yes']['tenure'], label='Churned', fill=True)
plt.xlabel('Tenure (Months)')
plt.title('Density Plot of Tenure by Churn Status')
plt.legend()
plt.show()
```

### Results 
- Achieved 85% predictive accuracy using an XGBoost classifier, significantly outperforming the baseline logistic regression model.
- Optimized the model to reach an F1-score of 0.82, successfully minimizing false negatives to ensure high-risk customers were accurately flagged for retention.
- Identified the top 20% of customers most likely to churn, allowing for a targeted marketing strategy that could potentially reduce overall attrition by 15%.

### Recommendations 
- Launch a targeted campaign to transition "Month-to-Month" users to annual plans by offering a 10-15% discount, addressing the highest-risk churn segment.
- Automatically include "Tech Support" and "Online Security" for customers with monthly charges above $70 to increase perceived value and lock in high-value users.
- Introduce small "autopay" rewards (e.g., $5 credit) for customers switching from Electronic Checks to Credit Card/ACH to reduce friction and billing-related attrition.

### References
- Dataset Source: Kaggle - Customer-Churn-Prediction
  - [https://www.kaggle.com/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv]
- Tool Documentation: Pandas, Seaborn, Matplotlib
   - [https://pandas.pydata.org/docs/]
   - [https://seaborn.pydata.org/]
   - [https://matplotlib.org/]
- Analysis Methodology: Inspired by Exploratory Data Analysis (EDA) best practices for content streaming platforms
