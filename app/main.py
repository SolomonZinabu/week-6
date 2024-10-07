from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel
import scorecardpy as sc  # Make sure this is imported
import numpy as np
# Initialize FastAPI app
app = FastAPI()

# Load your trained models
with open("./models/logistic_regression_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("./models/random_forest_model.pkl", "rb") as f:
    random_forest_model = pickle.load(f)

# Define a function to apply WoE


def apply_woe(data):
    # Load WoE bins from the saved model
    with open("./models/woe_bins.pkl", "rb") as f:
        woe_bins = pickle.load(f)
    
    # Convert woe_bins to a DataFrame if it’s an array
    if isinstance(woe_bins, np.ndarray):
        woe_bins = pd.DataFrame(woe_bins)

    # Assuming the first column contains the variable names
    if 'variable' in woe_bins.columns:
        required_columns = list(woe_bins['variable'].unique())
    else:
        # If the column does not exist, we assume the first column is the one we need
        required_columns = list(woe_bins.iloc[:, 0].unique())
    
    print("Required Columns:", required_columns)

    # Ensure that the input data has the required columns
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    return sc.woebin_ply(data, woe_bins)


# Input data format
class PredictionInput(BaseModel):
    transaction_hour: float
    transaction_day: float
    transaction_month: float
    transaction_year: int
    # Add more features as required

# Endpoint for logistic regression prediction
@app.post("/predict/logistic")
async def predict_logistic(input_data: PredictionInput):
    data = pd.DataFrame([input_data.dict()])
    data_woe = apply_woe(data)  # Apply WoE here
    prediction = logistic_model.predict(data_woe)
    return {"prediction": int(prediction[0])}

# Endpoint for random forest prediction
@app.post("/predict/randomforest")
async def predict_random_forest(input_data: PredictionInput):
    data = pd.DataFrame([input_data.dict()])
    data_woe = apply_woe(data)  # Apply WoE here
    prediction = random_forest_model.predict(data_woe)
    return {"prediction": int(prediction[0])}
