from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and preprocessing objects
model = joblib.load("disease_prediction_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
all_symptoms = joblib.load("symptom_list.pkl")

class SymptomRequest(BaseModel):
    symptoms: List[str]

@app.post("/diagnose")
async def diagnose(request: SymptomRequest):
    # Create input vector (1 for present symptoms, 0 for absent)
    input_vector = [1 if symptom in request.symptoms else 0 for symptom in all_symptoms]
    
    # Scale the input vector
    input_scaled = scaler.transform([input_vector])
    
    # Make prediction
    prediction_encoded = model.predict(input_scaled)[0]
    
    # Decode the prediction to get disease name
    disease_name = label_encoder.inverse_transform([prediction_encoded])[0]
    
    return {
        "disease": disease_name,
        "confidence": 1.0  # You can add confidence score if needed
    } 