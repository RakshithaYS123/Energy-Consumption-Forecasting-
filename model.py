import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Load Dataset
def load_data(file_path='Energy_consumption.csv'):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        sys.exit(f"Error: Dataset file '{file_path}' not found.")
    
    # Process timestamp if available
    if 'Timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['Timestamp'])
        df['month'] = df['date'].dt.month
        df['dayofweek'] = df['date'].dt.dayofweek
        df['hour'] = df['date'].dt.hour
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df.drop(['Timestamp', 'date'], axis=1, inplace=True)
    
    # Map categorical columns
    mappings = {
        'HVACUsage': {'On': 1, 'Off': 0}, 
        'LightingUsage': {'On': 1, 'Off': 0}, 
        'Holiday': {'Yes': 1, 'No': 0}
    }
    
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    # Encode day of week if exists
    if 'DayOfWeek' in df.columns:
        df['DayOfWeek'] = LabelEncoder().fit_transform(df['DayOfWeek'].astype(str))
    
    # Create feature interactions
    if 'Temperature' in df.columns and 'RenewableEnergy' in df.columns:
        df['temp_diff'] = df['Temperature'] - df['RenewableEnergy']
    
    # Fill missing values
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Check for target variable
    if 'EnergyConsumption' not in df.columns:
        sys.exit("Error: Target variable 'EnergyConsumption' not found.")
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# Train and Evaluate Models
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    print("Training models...")
    results = []
    
    for model, name in models:
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append((name, model, mse, r2))
        
        # Save model
        with open(f'{name.replace(" ", "_").lower()}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"  {name} - MSE: {mse:.4f}, R²: {r2:.4f}")
        
    return results

# Main Execution
if __name__ == "__main__":
    print("Starting model training process...")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Change to models directory
    os.chdir(models_dir)
    
    # Check for dataset file
    data_file = '../Energy_consumption.csv'
    if not os.path.exists(data_file):
        print(f"Warning: '{data_file}' not found. Please place your dataset file in the correct location.")
        data_file = input("Enter the path to your dataset file: ")
    
    # Load and process data
    df = load_data(data_file)
    
    # Prepare data for training
    X = df.drop('EnergyConsumption', axis=1)
    y = df['EnergyConsumption']
    
    print(f"Features: {', '.join(X.columns)}")
    print(f"Target: EnergyConsumption")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Define models
    models = [
        (LinearRegression(), "Linear Regression"),
        (Ridge(alpha=1.0), "Ridge Regression"),
        (Lasso(alpha=0.1), "Lasso Regression"),
        (DecisionTreeRegressor(max_depth=10), "Decision Tree Regressor"),
        (RandomForestRegressor(n_estimators=100), "Random Forest Regressor"),
        (XGBRegressor(n_estimators=100, learning_rate=0.1), "XGBoost Regressor")
    ]
    
    # Train and evaluate models
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    
    # Sort results by MSE (lower is better)
    results.sort(key=lambda x: x[2])
    
    print("\nModel Comparison Results:")
    print("--------------------------")
    for name, _, mse, r2 in results:
        print(f"{name}: MSE = {mse:.4f}, R² = {r2:.4f}")
    
    best_model = results[0]
    print(f"\nBest Model: {best_model[0]} with MSE: {best_model[2]:.4f} and R²: {best_model[3]:.4f}")
    
    # Save model metrics for application
    metrics = {}
    for name, _, mse, r2 in results:
        key = name.replace(" ", "_").lower()
        metrics[key] = {"name": name, "mse": mse, "r2": r2}
    
    with open("model_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    
    print("Model training complete. Models saved to 'models' directory.")