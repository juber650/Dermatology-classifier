from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np

BASE_DIR = Path(__file__).parent

# Initialize the FastAPI app
app = FastAPI(
    title="Dermatology Disease Classifier",
    description="API to predict 6 classes of skin disease based on 34 clinical features."
)

# Load the trained Random Forest model at startup
try:
    model = joblib.load(BASE_DIR / 'random_forest_dermatology_model.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Define the expected JSON payload schema
class PatientData(BaseModel):
    # Expects exactly 34 values: 33 clinical/histopathological features (0-3) and 1 Age (raw)
    features: list[float]

# Dictionary mapping class numbers to readable disease names
DISEASE_MAP = {
    1: "Psoriasis",
    2: "Seboreic Dermatitis",
    3: "Lichen Planus",
    4: "Pityriasis Rosea",
    5: "Chronic Dermatitis",
    6: "Pityriasis Rubra Pilaris"
}

# Serve the frontend at the root URL
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "index.html")

@app.post("/predict")
def predict_disease(data: PatientData):
    if len(data.features) != 34:
        raise HTTPException(status_code=400, detail="Exactly 34 features are required.")

       # Convert input list to a 2D numpy array for the model
    input_data = np.array(data.features).reshape(1, -1)

    # The model was trained with BOTH 'Age' (raw, index 33) and 'Age_scaled' (index 34),
    # so keep the raw age and append the scaled value as a 35th column.
    raw_age = input_data[0, 33]
    age_scaled = (raw_age - 0.0) / (75.0 - 0.0) * 3.0
    input_data = np.append(input_data, age_scaled).reshape(1, -1)

    # Generate prediction and probability scores
    predicted_class = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities) * 100

    return {
        "class_id": predicted_class,
        "diagnosis": DISEASE_MAP.get(predicted_class, "Unknown"),
        "confidence_score": f"{confidence:.2f}%",
        "all_probabilities": {DISEASE_MAP[i+1]: f"{prob*100:.1f}%" for i, prob in enumerate(probabilities)}
    }