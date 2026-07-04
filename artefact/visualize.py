import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_simulated_data(df, output_dir="."):
    """
    Plots a sample of the simulated energy consumption data (e.g., first 7 days).
    """
    plt.figure(figsize=(12, 6))
    
    # Take a 7-day slice (168 hours)
    sample_df = df.head(168)
    
    plt.plot(sample_df.index, sample_df['total_consumption_kwh'], label='Total Consumption (kWh)', color='blue')
    plt.plot(sample_df.index, sample_df['temperature_celsius'] / 10, label='Temperature (Scaled)', color='orange', alpha=0.5)
    
    plt.title('Simulated Smart Home Energy Consumption (1-Week Sample)')
    plt.xlabel('Date')
    plt.ylabel('Consumption (kWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, '1_simulated_data_sample.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved graph: {filepath}")

def plot_feature_importance(model, feature_names, output_dir="."):
    """
    Plots the feature importance of the trained Random Forest model.
    """
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)
    
    plt.barh(range(len(indices)), importances[indices], color='green', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title('AI Model Feature Importance for Energy Prediction')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, '2_feature_importance.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved graph: {filepath}")

def plot_actual_vs_predicted(model, X_test, y_test, output_dir="."):
    """
    Plots Actual vs Predicted energy consumption on the test set.
    """
    predictions = model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    
    # Plot first 100 predictions to avoid clutter
    subset_size = min(100, len(y_test))
    
    plt.plot(range(subset_size), y_test.values[:subset_size], label='Actual Consumption', color='blue', marker='o', markersize=4)
    plt.plot(range(subset_size), predictions[:subset_size], label='Predicted Consumption', color='red', linestyle='--', marker='x', markersize=4)
    
    plt.title('AI Prediction Accuracy: Actual vs. Predicted (Test Set Sample)')
    plt.xlabel('Test Samples (Hours)')
    plt.ylabel('Energy Consumption (kWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, '3_actual_vs_predicted.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved graph: {filepath}")

def plot_future_predictions(future_df, predictions, current_hour, output_dir="."):
    """
    Plots the real-time future predictions with peak hours highlighted.
    """
    plt.figure(figsize=(10, 6))
    
    hours = future_df['hour'].values
    
    # Create x-axis labels
    x_labels = [f"{h}:00" for h in hours]
    x_pos = np.arange(len(hours))
    
    # Determine colors based on peak hours (16-20)
    colors = ['red' if 16 <= h <= 20 else 'green' for h in hours]
    
    bars = plt.bar(x_pos, predictions, color=colors)
    plt.xticks(x_pos, x_labels)
    
    plt.title('Predicted Energy Consumption for the Next 6 Hours')
    plt.xlabel('Hour of Day')
    plt.ylabel('Predicted Consumption (kWh)')
    
    # Add a legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Peak Hours (Expensive)'),
                       Patch(facecolor='green', label='Off-Peak Hours')]
    plt.legend(handles=legend_elements)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}", ha='center', va='bottom')
        
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, '4_future_predictions_optimization.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved graph: {filepath}")
