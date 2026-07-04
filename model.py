import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pickle

def prepare_features(df):
    """
    Prepare features for model training.
    """
    X = df[['hour', 'day_of_week', 'is_weekend', 'temperature_celsius', 'occupancy']]
    y = df['total_consumption_kwh']
    return X, y

def train_model(df):
    """
    Train a Random Forest model to predict energy consumption.
    """
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"Model Evaluation - MAE: {mae:.3f} kWh, RMSE: {rmse:.3f} kWh")
    
    # Save model
    with open('energy_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    return model, X_test, y_test

def predict_future(model, future_features):
    """
    Predict energy consumption for future hours.
    """
    return model.predict(future_features)
