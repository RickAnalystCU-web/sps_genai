import io
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from helper_lib.data_loader import CIFAR10_CLASSES
from helper_lib.model import get_model
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
import spacy

app = FastAPI(
    title="SPS GenAI Embedding API",
    description="A simple FastAPI app that uses spaCy to generate text embeddings and compare semantic similarity.",
    version="1.0.0",
)
device = "cuda" if torch.cuda.is_available() else "cpu"

image_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)

classifier_model = get_model("CNN", num_classes=10)
checkpoint_path = Path("checkpoints") / "cnn_cifar10.pth"

if checkpoint_path.exists():
    classifier_model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    classifier_model.to(device)
    classifier_model.eval()
else:
    classifier_model = None
nlp = spacy.load("en_core_web_lg")


class SimilarityRequest(BaseModel):
    text1: str
    text2: str


@app.get("/")
def read_root():
    return {
        "message": "SPS GenAI Embedding API is running.",
        "endpoints": {
            "embedding": "/embedding/{text}",
            "similarity": "/similarity",
            "docs": "/docs",
        },
    }


@app.get("/embedding/{text}")
def get_embedding(text: str):
    doc = nlp(text)

    return {
        "text": text,
        "has_vector": bool(doc.has_vector),
        "vector_dimension": len(doc.vector),
        "vector_norm": float(doc.vector_norm),
        "embedding_preview": [float(value) for value in doc.vector[:10]],
    }


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    doc1 = nlp(request.text1)
    doc2 = nlp(request.text2)

    return {
        "text1": request.text1,
        "text2": request.text2,
        "text1_has_vector": bool(doc1.has_vector),
        "text2_has_vector": bool(doc2.has_vector),
        "similarity": float(doc1.similarity(doc2)),
    }
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict the CIFAR10 class of an uploaded image.
    """

    if classifier_model is None:
        raise HTTPException(
            status_code=503,
            detail="CNN model checkpoint not found. Run train_cifar10.py first.",
        )

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classifier_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_index = int(predicted_index.item())
    confidence = float(confidence.item())

    return {
        "filename": file.filename,
        "predicted_class": CIFAR10_CLASSES[predicted_index],
        "class_index": predicted_index,
        "confidence": confidence,
        "device": device,
    }