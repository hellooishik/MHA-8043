from data_generation import generate_smart_home_data
from model import train_model
from optimize import optimize_energy_usage
import visualize
import pandas as pd
from datetime import datetime

def main():
    print("--- Smart Home Energy Prediction and Optimization System ---")
    
    # 1. Generate Data
    print("\n1. Collecting historical smart home data...")
    df = generate_smart_home_data(days=60) # 60 days of data
    print(f"   Collected {len(df)} hours of historical data.")
    
    # Generate graph 1
    visualize.plot_simulated_data(df)
    
    # 2. Train Model
    print("\n2. Training AI prediction model...")
    model, X_test, y_test = train_model(df)
    
    # Generate graphs 2 and 3
    feature_names = ['hour', 'day_of_week', 'is_weekend', 'temperature_celsius', 'occupancy']
    visualize.plot_feature_importance(model, feature_names)
    visualize.plot_actual_vs_predicted(model, X_test, y_test)
    
    # 3. Simulate a real-time prediction and optimization scenario
    print("\n3. Running real-time prediction for the next 6 hours...")
    
    # Let's say it's currently 14:00 (2 PM) on a weekday
    current_hour = 14
    future_data = []
    for i in range(1, 7):
        hour = (current_hour + i) % 24
        future_data.append({
            'hour': hour,
            'day_of_week': 2, # Wednesday
            'is_weekend': 0,
            'temperature_celsius': 18.0 - (i * 0.5), # temp dropping in evening
            'occupancy': 1 if hour >= 17 else 0 # people coming home
        })
        
    future_df = pd.DataFrame(future_data)
    
    # Get predictions
    future_features = future_df[['hour', 'day_of_week', 'is_weekend', 'temperature_celsius', 'occupancy']]
    predictions = model.predict(future_features)
    
    print("\nPredicted Energy Consumption (next 6 hours):")
    for i, pred in enumerate(predictions):
        h = (current_hour + i + 1) % 24
        print(f"   {h}:00 -> {pred:.2f} kWh")
        
    # Generate graph 4
    visualize.plot_future_predictions(future_df, predictions, current_hour)
        
    # 4. Optimization Recommendations
    print("\n4. Generating Optimization Recommendations...")
    recommendations = optimize_energy_usage(predictions, current_hour)
    
    if not recommendations:
        print("   Usage looks optimal. No recommendations at this time.")
    else:
        for rec in recommendations:
            print(f"   * {rec}")
            
    print("\n--- System Execution Complete ---")

if __name__ == "__main__":
    main()
