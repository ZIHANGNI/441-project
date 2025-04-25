import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def load_and_preprocess():
    df = pd.read_csv("C:/Users/nzh00/Desktop/2025 SPRING/441/project/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    df.drop('customerID', axis=1, inplace=True)

    binary_map = {'Yes': 1, 'No': 0,
                  'No phone service': 0,
                  'No internet service': 0}
    binary_cols = ['Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'Churn', 'MultipleLines',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies']

    for col in binary_cols:
        df[col] = df[col].replace(binary_map)

    cat_cols = ['gender', 'InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df.apply(pd.to_numeric, errors='ignore')


def eda_analysis(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Churn', data=df)
    plt.title('Customer Churn Distribution')
    plt.show()

    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(16, 12))
    corr = numeric_df.corr()
    sns.heatmap(corr[abs(corr['Churn']) > 0.05], annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation with Churn (|corr| > 0.05)')
    plt.show()


def train_model(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    joblib.dump(scaler, 'scaler.pkl')

    print("\nLogistic Regression Performance:")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)
    print(classification_report(y_test, lr.predict(X_test)))

    print("\nRandom Forest Tuning...")
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    print("\nBest Random Forest Performance:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(classification_report(y_test, best_rf.predict(X_test)))

    feature_importance = best_rf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
    plt.title('Top 10 Important Features')
    plt.show()

    joblib.dump(best_rf, 'churn_model.pkl')
    return best_rf, scaler


if __name__ == "__main__":
    df = load_and_preprocess()

    eda_analysis(df)

    model, scaler = train_model(df)

    print("\nModel training completed. Files saved:")
    print("- churn_model.pkl (Trained model)")
    print("- scaler.pkl (Feature scaler)")
