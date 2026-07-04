# AI-Powered Smart Home Energy Consumption Prediction and Optimization System

This project is a working artifact for the AI Systems Engineering assignment. It implements a key subsystem of an AI-powered smart home that predicts energy consumption and provides optimization recommendations for sustainable residential buildings in the UK.

## Features
1. **Data Simulation**: Generates synthetic, realistic time-series data for a smart home, including features like temperature, occupancy, and time of day.
2. **Machine Learning**: Uses a Random Forest Regressor to predict future energy consumption based on historical patterns.
3. **Optimization Strategy**: Analyzes predicted consumption against UK time-of-use tariffs (peak/off-peak hours) and suggests actionable energy-saving behaviors.

## Project Structure
- `data_generation.py`: Contains functions to generate synthetic smart home dataset.
- `model.py`: Handles feature preparation, model training, evaluation, and saving/loading the model.
- `optimize.py`: Contains logic to provide energy shifting recommendations based on predictions.
- `main.py`: The main entry point script that orchestrates the entire pipeline.
- `requirements.txt`: Python package dependencies.

## Setup Instructions

1. Ensure you have Python 3.8+ installed.
2. Navigate to the project directory:
   ```bash
   cd artefact
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

Execute the main script to run the complete simulation:

```bash
python main.py
```

### Output Expectation
The script will output:
- The amount of simulated historical data generated.
- Model evaluation metrics (MAE and RMSE).
- Predicted energy consumption for the next 6 hours.
- Actionable recommendations to shift energy load away from peak hours, aligning with sustainable practices.
