import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import numpy as np

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the input data model using Pydantic
class AQIInput(BaseModel):
    datetime_str: str 
    PM2_5: float
    PM10: float
    O3: float
    NO2: float
    SO2: float
    CO: float
    Temperature: float
    Humidity: float
    Wind_Speed: float
    Pressure: float
    AQI_Lag1: float

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
async def predict_aqi(data: AQIInput):
    # Convert input data to a dictionary
    input_dict = data.dict()

    # Parse datetime string and extract temporal features
    dt_object = datetime.fromisoformat(input_dict.pop('datetime_str'))
    input_dict['Hour'] = dt_object.hour
    input_dict['DayOfWeek'] = dt_object.weekday() # Monday=0, Sunday=6
    input_dict['Month'] = dt_object.month
    input_dict['Year'] = dt_object.year

    # Calculate derived ratio features
    input_dict['PM2_5_PM10_Ratio'] = input_dict['PM2_5'] / (input_dict['PM10'] + 1e-6)
    input_dict['NO2_SO2_Ratio'] = input_dict['NO2'] / (input_dict['SO2'] + 1e-6)

    # The order of features needs to match the training data
    features_order = [
        'PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO', 'Temperature', 'Humidity', 'Wind_Speed', 'Pressure',
        'Hour', 'DayOfWeek', 'Month', 'Year', 'PM2.5_PM10_Ratio', 'NO2_SO2_Ratio', 'AQI_Lag1'
    ]

    # Create a DataFrame from the processed input dictionary
    processed_input_data = {
        'PM2.5': input_dict['PM2_5'],
        'PM10': input_dict['PM10'],
        'O3': input_dict['O3'],
        'NO2': input_dict['NO2'],
        'SO2': input_dict['SO2'],
        'CO': input_dict['CO'],
        'Temperature': input_dict['Temperature'],
        'Humidity': input_dict['Humidity'],
        'Wind_Speed': input_dict['Wind_Speed'],
        'Pressure': input_dict['Pressure'],
        'Hour': input_dict['Hour'],
        'DayOfWeek': input_dict['DayOfWeek'],
        'Month': input_dict['Month'],
        'Year': input_dict['Year'],
        'PM2.5_PM10_Ratio': input_dict['PM2_5_PM10_Ratio'],
        'NO2_SO2_Ratio': input_dict['NO2_SO2_Ratio'],
        'AQI_Lag1': input_dict['AQI_Lag1']
    }

    input_df = pd.DataFrame([processed_input_data], columns=features_order)

    # Scale the input data
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    return {"predicted_aqi": prediction.item()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)