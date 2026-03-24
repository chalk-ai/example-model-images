"""
FastAPI server to host a classifier/regressor model for scaling groups
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List, Optional, Union

app = FastAPI()

# Load the model once at startup
model = None
feature_names = None


@app.on_event("startup")
async def startup():
    global model, feature_names
    print("Loading classifier/regressor model...")
    # Load a pre-trained model (you can swap this with your own trained model)
    # Example: model = joblib.load("/app/model.pkl")
    try:
        model = joblib.load("/app/model.pkl")
        # Optionally load feature names if available
        try:
            feature_names = joblib.load("/app/feature_names.pkl")
        except FileNotFoundError:
            feature_names = None
        print("Model loaded!")
    except FileNotFoundError:
        print("WARNING: model.pkl not found. Using a dummy model for testing.")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(np.random.rand(10, 4), np.random.randint(0, 2, 10))


class PredictRequest(BaseModel):
    """Input features for prediction"""
    features: List[float]


class PredictResponse(BaseModel):
    """Prediction output"""
    prediction: Optional[float] = None
    probabilities: Optional[List[float]] = None
    class_labels: Optional[List[str]] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: str


@app.post("/predict")
async def predict(request: PredictRequest) -> Union[PredictResponse, ErrorResponse]:
    """Make a prediction using the model"""
    if model is None:
        return ErrorResponse(error="Model not loaded")

    try:
        X = np.array(request.features).reshape(1, -1)
        prediction = model.predict(X)[0]

        response = PredictResponse(prediction=float(prediction))

        # If the model has predict_proba (classifier), include probabilities
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)[0]
            response.probabilities = [float(p) for p in probas]
            if hasattr(model, "classes_"):
                response.class_labels = [str(c) for c in model.classes_]

        return response
    except Exception as e:
        return ErrorResponse(error=str(e))


@app.post("/predict_proba")
async def predict_proba(request: PredictRequest) -> Union[PredictResponse, ErrorResponse]:
    """Get class probabilities (for classifiers)"""
    if model is None:
        return ErrorResponse(error="Model not loaded")

    if not hasattr(model, "predict_proba"):
        return ErrorResponse(error="Model does not support predict_proba")

    try:
        X = np.array(request.features).reshape(1, -1)
        probas = model.predict_proba(X)[0]

        response = PredictResponse(
            probabilities=[float(p) for p in probas],
        )
        if hasattr(model, "classes_"):
            response.class_labels = [str(c) for c in model.classes_]

        return response
    except Exception as e:
        return ErrorResponse(error=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
