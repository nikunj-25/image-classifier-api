from fastapi import FastAPI, UploadFile, File
from app.model import predict
from app.schemas import PredictionResponse
from app.utils import preprocess_image

app = FastAPI()

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)
    label = predict(image_tensor)
    return PredictionResponse(label=label)