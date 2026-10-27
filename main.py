
import os
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="House Price Prediction API",
    description="TensorFlow-powered House Price Prediction Service",
    version="1.0.0"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "house_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

class HouseFeatures(BaseModel):

    med_inc: float = Field(gt=0, le=20)
    house_age: float = Field(gt=0, le=100)
    ave_rooms: float = Field(gt=0, le=100)
    ave_bedrms: float = Field(gt=0, le=20)
    population: float = Field(gt=0, le=50000)
    ave_occup: float = Field(gt=0, le=100)
    latitude: float = Field(ge=32, le=42)
    longitude: float = Field(ge=-125, le=-114)

@app.get("/")
def home():

    return {
        "message": "House Price Prediction API is running!",
        "status": "success",
        "documentation": "/docs"
    }

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/predict")
def predict_price(data: HouseFeatures):

    try:

        features = np.array([[
            data.med_inc,
            data.house_age,
            data.ave_rooms,
            data.ave_bedrms,
            data.population,
            data.ave_occup,
            data.latitude,
            data.longitude
        ]], dtype=np.float32)

        prediction = model.predict(
            features,
            verbose=0
        )

        predicted_value = float(prediction[0][0])
        predicted_value = max(0.0, predicted_value)

        price_usd = predicted_value * 100000

        return {
            "status": "success",
            "predicted_price_100k_usd": round(
                predicted_value, 4
            ),
            "predicted_price_usd": round(
                price_usd, 2
            ),
            "currency": "USD"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )