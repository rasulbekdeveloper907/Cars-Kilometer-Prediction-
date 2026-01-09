import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Model path (Docker + Windows + Linux compatible)
# --------------------------------------------------
MODEL_PATH = Path("Models") / "Pipeline_Models" / "RandomForestRegressor_Fast.joblib"

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="Cars Kilometer Prediction API",
    version="1.0"
)

pipeline = None

# --------------------------------------------------
# Load model on startup
# --------------------------------------------------
@app.on_event("startup")
def load_model():
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        pipeline = None

# --------------------------------------------------
# Schemas
# --------------------------------------------------
class DatasetInput(BaseModel):
    index: int
    dateCrawled: str
    name: str
    seller: str
    offerType: str
    price: int
    abtest: str
    vehicleType: str
    yearOfRegistration: int
    gearbox: str
    powerPS: int
    model: str
    kilometer: int
    monthOfRegistration: int
    fuelType: str
    brand: str
    notRepairedDamage: str
    dateCreated: str
    nrOfPictures: int
    postalCode: int
    lastSeen: str

class PredictionOutput(BaseModel):
    predicted_cluster: int
    cluster_probability: float

# --------------------------------------------------
# Health endpoints
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------------------------------------
# Predict endpoint
# --------------------------------------------------
@app.post("/predict", response_model=PredictionOutput)
def predict(data: DatasetInput):

    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Pydantic v2 compatible
    df = pd.DataFrame([data.model_dump()])

    try:
        proba_all = pipeline.predict_proba(df)[0]
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="Model does not support predict_proba()"
        )

    predicted_cluster = int(proba_all.argmax())
    cluster_probability = float(proba_all[predicted_cluster])

    logger.info(
        "Predicted cluster=%s probability=%.4f",
        predicted_cluster,
        cluster_probability
    )

    return PredictionOutput(
        predicted_cluster=predicted_cluster,
        cluster_probability=round(cluster_probability, 4)
    )
