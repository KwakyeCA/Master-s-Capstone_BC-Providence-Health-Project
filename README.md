# ED Wait Time Prediction Model

## Project Overview
Emergency Department (ED) overcrowding and prolonged wait times are persistent challenges for healthcare systems. Accurate forecasting of patient admissions can support better staffing decisions, resource allocation, and operational planning.

This project focuses on **predicting daily patient admissions from the MSJ Emergency Department** using historical ED data, admission rates, complaint codes, and contextual variables such as seasonality and weather. The predicted admissions serve as a proxy for anticipating ED wait time pressure and congestion.

The project adopts a **regression-based analytical approach**, emphasizing interpretability and real-world applicability rather than complex machine learning pipelines.

---

## Objectives
- Model and predict **daily ED admissions**
- Identify key factors influencing ED demand
- Quantify the impact of temporal, clinical, and environmental variables
- Provide an interpretable predictive framework to support operational decision-making

---

##  Data Description
The dataset consists of historical Emergency Department records and includes:

- **Temporal variables**
  - Date
  - Day of week
  - Month
  - Seasonal indicators

- **Clinical variables**
  - Admission counts
  - Complaint codes
  - Admission rates

- **Contextual variables**
  - Weather indicators
  - Seasonal trends

All data were preprocessed to handle missing values, outliers, and temporal dependencies.

---

## Methodology
The analytical workflow follows a structured regression modeling pipeline:

1. **Exploratory Data Analysis (EDA)**
   - Trend and seasonality analysis
   - Distribution of admissions
   - Correlation analysis

2. **Feature Engineering**
   - Encoding categorical variables
   - Seasonal and calendar-based features
   - Aggregation of complaint codes

3. **Model Development**
   - Multiple Linear Regression
   - Model diagnostics and assumption checks
   - Variable selection and interpretation of coefficients

4. **Model Evaluation**
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Residual analysis

---

## Key Results
- The model captures **seasonal and temporal patterns** in ED admissions effectively.
- Calendar effects (day of week and seasonality) are strong predictors of ED demand.
- Contextual factors such as weather show moderate but meaningful influence.
- The regression framework provides **transparent and explainable insights** suitable for healthcare stakeholders.

---

## Tools & Technologies
- **Python**
- Pandas & NumPy (data manipulation)
- Matplotlib & Seaborn (visualization)
- Statsmodels / Scikit-learn (regression modeling)

---

## Practical Applications
- Anticipating high-demand ED days
- Supporting staffing and shift planning
- Informing short-term capacity management
- Enhancing patient flow and reducing wait times

---

## Future Improvements
- Incorporate real-time ED arrival data
- Extend the model to directly predict **wait times** instead of admissions
- Compare regression results with tree-based or ensemble models
- Integrate hospital capacity and staffing data

---

## Project Status
Completed as part of an academic data analytics project (Capstone), with scope for extension into operational healthcare analytics.

---

## Co-Author
**Cosmos Ameyaw Kwakye**  
Master’s Student – Data Analytics Engineering  
Focus Areas: Applied Regression, Performance Analytics, Healthcare Analytics

---

## Disclaimer
This project is for academic and analytical purposes only. Results should not be used as the sole basis for clinical or operational decisions.
