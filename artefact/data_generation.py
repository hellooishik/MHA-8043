import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_smart_home_data(days=30):
    """
    Generate synthetic smart home energy consumption data.
    """
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=days)
    
    # Hourly data
    hours = days * 24
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    
    data = []
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.weekday()
        
        # Base load (fridge, standby devices)
        base_load = 0.5 
        
        # Temperature (simulated sinusoidal with noise)
        temperature = 15 + 10 * np.sin(hour * np.pi / 12) + np.random.normal(0, 2)
        
        # Occupancy (higher probability in evenings and weekends)
        if day_of_week >= 5: # Weekend
            occupancy = 1 if np.random.rand() > 0.3 else 0
        else: # Weekday
            if 9 <= hour <= 17:
                occupancy = 1 if np.random.rand() > 0.8 else 0
            else:
                occupancy = 1 if np.random.rand() > 0.2 else 0
                
        # HVAC load (cooling if temp > 22, heating if temp < 18)
        hvac_load = 0
        if temperature > 22 and occupancy:
            hvac_load = (temperature - 22) * 0.5
        elif temperature < 18 and occupancy:
            hvac_load = (18 - temperature) * 0.8
            
        # Appliance load (cooking, TV, etc.)
        appliance_load = 0
        if occupancy:
            if 7 <= hour <= 9 or 18 <= hour <= 21:
                appliance_load = np.random.normal(2.0, 0.5)
            else:
                appliance_load = np.random.normal(0.5, 0.2)
                
        appliance_load = max(0, appliance_load)
        
        # Total consumption in kWh
        total_consumption = base_load + hvac_load + appliance_load + np.random.normal(0, 0.1)
        total_consumption = max(0.1, total_consumption) # ensure positive
        
        data.append({
            'timestamp': ts,
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'temperature_celsius': round(temperature, 2),
            'occupancy': occupancy,
            'total_consumption_kwh': round(total_consumption, 3)
        })
        
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

if __name__ == "__main__":
    df = generate_smart_home_data()
    df.to_csv("smart_home_data.csv")
    print(f"Generated {len(df)} records of synthetic data.")
