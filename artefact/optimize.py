import pandas as pd

def optimize_energy_usage(predictions, current_hour):
    """
    Suggests optimizations based on predicted energy usage.
    Assumes time-of-use (TOU) tariffs where peak hours are more expensive.
    In the UK, typical peak hours might be 16:00 - 20:00.
    """
    peak_hours = [16, 17, 18, 19, 20]
    off_peak_hours = [0, 1, 2, 3, 4, 5, 6, 7]
    
    recommendations = []
    
    total_predicted_peak = 0
    
    for i, pred_kwh in enumerate(predictions):
        pred_hour = (current_hour + i + 1) % 24
        
        if pred_hour in peak_hours and pred_kwh > 2.0:
            total_predicted_peak += pred_kwh
            recommendations.append(
                f"Warning: High consumption ({pred_kwh:.2f} kWh) predicted during peak hour {pred_hour}:00. "
                "Recommendation: Delay using high-energy appliances (washing machine, dishwasher) until after 21:00."
            )
            
        elif pred_hour in off_peak_hours and pred_kwh < 1.0:
             recommendations.append(
                f"Opportunity: Low consumption predicted during off-peak hour {pred_hour}:00. "
                "Recommendation: Good time to schedule EV charging or run background appliances."
            )
             
    if total_predicted_peak > 5.0:
        recommendations.insert(0, "CRITICAL: High overall peak usage predicted. Consider battery storage discharge if available.")
        
    return recommendations
