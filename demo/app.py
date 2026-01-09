import joblib
import pandas as pd
import gradio as gr
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# Model path (Docker-safe)
# ======================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "Models" / "Pipeline_Models" / "RandomForestRegressor_Fast.joblib"

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    logger.error("Failed to load model: %s", e)
    model = None

# ======================
# Predict function
# ======================
def predict(
    dateCrawled,
    name,
    seller,
    offerType,
    price,
    abtest,
    vehicleType,
    yearOfRegistration,
    gearbox,
    powerPS,
    model_name,
    kilometer,
    monthOfRegistration,
    fuelType,
    brand,
    notRepairedDamage,
    dateCreated,
    nrOfPictures,
    postalCode,
    lastSeen
):
    if model is None:
        return {"error": "Model not loaded"}

    df = pd.DataFrame([{
        "dateCrawled": dateCrawled,
        "name": name,
        "seller": seller,
        "offerType": offerType,
        "price": int(price),
        "abtest": abtest,
        "vehicleType": vehicleType,
        "yearOfRegistration": int(yearOfRegistration),
        "gearbox": gearbox,
        "powerPS": int(powerPS),
        "model": model_name,
        "kilometer": int(kilometer),
        "monthOfRegistration": int(monthOfRegistration),
        "fuelType": fuelType,
        "brand": brand,
        "notRepairedDamage": notRepairedDamage,
        "dateCreated": dateCreated,
        "nrOfPictures": int(nrOfPictures),
        "postalCode": int(postalCode),
        "lastSeen": lastSeen
    }])

    predicted_cluster = int(model.predict(df)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba(df)[0]
        confidence = float(proba_all[predicted_cluster])

    return {
        "predicted_cluster": predicted_cluster,
        "confidence": round(confidence, 4) if confidence is not None else None
    }

# ======================
# Gradio UI
# ======================
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Date Crawled"),
        gr.Textbox(label="Name"),
        gr.Textbox(label="Seller"),
        gr.Textbox(label="Offer Type"),
        gr.Number(label="Price", precision=0),
        gr.Textbox(label="AB Test"),
        gr.Textbox(label="Vehicle Type"),
        gr.Number(label="Year Of Registration", precision=0),
        gr.Textbox(label="Gearbox"),
        gr.Number(label="Power PS", precision=0),
        gr.Textbox(label="Model"),
        gr.Number(label="Kilometer", precision=0),
        gr.Number(label="Month Of Registration", precision=0),
        gr.Textbox(label="Fuel Type"),
        gr.Textbox(label="Brand"),
        gr.Textbox(label="Not Repaired Damage"),
        gr.Textbox(label="Date Created"),
        gr.Number(label="Nr Of Pictures", precision=0),
        gr.Number(label="Postal Code", precision=0),
        gr.Textbox(label="Last Seen")
    ],
    outputs=gr.JSON(label="Prediction Result"),
    title="Car Kilometer Prediction ",
    description="RandomForestClassifier | Car Kilometer Prediction"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

