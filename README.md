# Customer Churn Prediction Based on Machine Learning

## Summary
This project aims to build a machine learning model to predict customer churn using a telecom industry dataset. The dataset includes features such as customer demographics, account tenure, monthly charges, and service usage patterns. The goal is to identify key drivers of churn and develop a predictive model that can help telecom companies improve customer retention strategies.

### Methodology
1. **Exploratory Data Analysis:** We analyze the dataset to understand feature distributions, correlations, and key patterns related to churn.
2. **Data Preprocessing:** Handling missing values, encoding categorical variables, and normalizing numerical data.
3. **Baseline Model:** Logistic Regression is used as the initial predictive model to establish a performance benchmark.
4. **Random Forest Model:** We train and optimize a Random Forest model, tuning hyperparameters to improve predictive accuracy.
5. **Evaluation:** Comparing model performance using accuracy, precision, recall, and F1-score, and analyzing feature importance.
6. **Future Improvements:** Exploring advanced algorithms like XGBoost and improving feature engineering techniques.

## Data Source
The dataset used in this project is the **Telco Customer Churn** dataset from Kaggle:  
[Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Packages Required
pandas
numpy
scikit-learn
streamlit
matplotlib
seaborn
joblib

## Instructions On How To Run The Code
1. Download dataset from Kaggle: Telco Customer Churn
2. Training the model by running 441_project.py
3. Launch prediction web app through https://498-project-oyzu5aneovu7kydzrbnice.streamlit.app/
4. Make Predictions:
    **Fill in customer details using the web form
    **Click "Predict Churn Risk"
    **View prediction and probability percentage
