from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Cargar modelo entrenado
model = joblib.load("app/pipeline.joblib")

# Nombres de las columnas usados durante el entrenamiento
COLUMN_NAMES = [
    'pregnancies', 'glucose', 'bloodpressure', 'skinthickness',
    'insulin', 'bmi', 'diabetespedigreefunction', 'age'
]


# Modelo de entrada
class InputData(BaseModel):
    features: list

# Endpoint raíz
@app.get("/")
def root():
    return {"message": "hola harold"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.features], columns=COLUMN_NAMES)
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}

@app.post("/predict_proba")
def predict_proba(data: InputData):
    df = pd.DataFrame([data.features], columns=COLUMN_NAMES)
    prediction_proba = model.predict_proba(df)
    return {"prediction_proba": prediction_proba.tolist()}
