from fastapi import FastAPI, UploadFile, File
from app.model import predict
from app.schemas import PredictionResponse
from app.utils import preprocess_image
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)
    label = predict(image_tensor)
    return PredictionResponse(label=label)

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")