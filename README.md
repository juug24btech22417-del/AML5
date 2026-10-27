# House Price Prediction API

A house price prediction system built using TensorFlow and FastAPI.

## Features
- TensorFlow Neural Network
- House Price Prediction
- FastAPI REST API
- Model Evaluation
- Public Deployment Ready

## API Endpoints

GET /
GET /health
POST /predict

## Run Locally

pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000

## API Documentation

http://localhost:8000/docs