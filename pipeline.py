import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def load_data(filepath):
    """Load the nitrogen dataset from a CSV file."""
    data = pd.read_csv(filepath)
    data.set_index('State', inplace=True)
    return data

def preprocess_data(data):
    """Handle missing values and perform data cleaning."""
    data.replace(-32767, 0, inplace=True)  # Placeholder values replaced with 0
    return data

def split_data(data):
    """Split the dataset into training and testing sets."""
    X = data.drop(columns=['Nitrogen gas - Master Comp Mass Frac (Nitrogen)'])
    y = data['Nitrogen gas - Master Comp Mass Frac (Nitrogen)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_baseline_model(X_train, y_train):
    """Train a baseline Linear Regression model."""
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    return baseline_model

def train_random_forest_model(X_train, y_train):
    """Train a Random Forest model."""
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance using MSE and R-squared."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def plot_feature_importances(rf_model, X_columns):
    """Plot feature importances of the Random Forest model."""
    importances = rf_model.feature_importances_
    feature_importance = pd.Series(importances, index=X_columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_importance.index, palette='viridis', dodge=False)
    plt.title('Feature Importances from Random Forest Model')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()

def main():
    """Main function to run the full workflow."""
    data = load_data('./data/nitrogen.csv')
    data = preprocess_data(data)
    
    print("Missing values after preprocessing:\n", data.isnull().sum())
    
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Baseline Model
    baseline_model = train_baseline_model(X_train, y_train)
    mse_baseline, r2_baseline = evaluate_model(baseline_model, X_test, y_test)
    print(f'Baseline Model - MSE: {mse_baseline}, R2: {r2_baseline}')
    
    # Random Forest Model
    rf_model = train_random_forest_model(X_train, y_train)
    mse_rf, r2_rf = evaluate_model(rf_model, X_test, y_test)
    print(f'Random Forest Model - MSE: {mse_rf}, R2: {r2_rf}')
    
    # Feature Importances
    plot_feature_importances(rf_model, X_train.columns)
    
    # Save the best model
    best_model = rf_model if r2_rf > r2_baseline else baseline_model
    joblib.dump(best_model, 'nitrogen_prediction_model.pkl')
    print(f'The best model is saved as nitrogen_prediction_model.pkl')

if __name__ == "__main__":
    main()