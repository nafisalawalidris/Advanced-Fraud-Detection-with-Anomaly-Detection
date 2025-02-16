from fastapi import FastAPI, HTTPException, Request
import uvicorn
import numpy as np
import joblib
from pydantic import BaseModel, validator
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the saved models and scaler
log_reg_model = joblib.load('log_reg_model.pkl')  # Logistic Regression model
rf_model = joblib.load('rf_model.pkl')           # Random Forest model
iso_forest_model = joblib.load('iso_forest_model.pkl')  # Isolation Forest model
oc_svm_model = joblib.load('oc_svm_model.pkl')   # One-Class SVM model
scaler = joblib.load('scaler.pkl')               # Scaler for preprocessing

# Define a Pydantic model for input validation
class TransactionData(BaseModel):
    features: List[float]  # List of 30 features (V1-V28, Amount, Time)

    @validator('features')
    def check_length(cls, v):
        if len(v) != 30:
            raise ValueError('Exactly 30 features are required')
        return v

# Create a FastAPI app
app = FastAPI()

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Fraud Detection API!",
        "usage": "Send a POST request to /predict with transaction data to get a fraud prediction."
    }

# Prediction endpoint
@app.post("/predict")
async def predict(transaction: TransactionData, request: Request):
    """
    Predict whether a transaction is fraudulent.
    
    Parameters:
        transaction: A JSON object containing the transaction features.
    
    Returns:
        A JSON object with the prediction (0 for non-fraudulent, 1 for fraudulent).
    """
    try:
        # Log the incoming request payload
        logger.info(f"Incoming request payload: {await request.json()}")

        # Convert input data to a numpy array
        input_data = np.array(transaction.features).reshape(1, -1)
        
        # Exclude the first feature (Time) to match the scaler's expectations
        input_data = input_data[:, 1:]  # Remove the first feature (Time)
        
        # Scale the input data using the pre-trained scaler
        input_data_scaled = scaler.transform(input_data)
        
        # Get predictions from supervised models
        log_reg_pred = log_reg_model.predict(input_data_scaled)
        rf_pred = rf_model.predict(input_data_scaled)
        
        # Get predictions from anomaly detection models
        iso_forest_pred = iso_forest_model.predict(input_data_scaled)
        iso_forest_pred = [1 if x == -1 else 0 for x in iso_forest_pred]  # Convert anomalies (-1) to 1 (fraud)
        
        oc_svm_pred = oc_svm_model.predict(input_data_scaled)
        oc_svm_pred = [1 if x == -1 else 0 for x in oc_svm_pred]  # Convert anomalies (-1) to 1 (fraud)
        
        # Perform majority voting
        ensemble_pred = np.round(np.mean([log_reg_pred, rf_pred, iso_forest_pred, oc_svm_pred]))
        
        # Return the prediction
        return {"prediction": int(ensemble_pred[0])}
    
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred")

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)