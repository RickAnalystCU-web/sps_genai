import base64
import io
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from helper_lib.data_loader import CIFAR10_CLASSES
from helper_lib.diffusion_model import (
    CIFAR10Diffusion,
    tensor_to_png_bytes as diffusion_tensor_to_png_bytes,
)
from helper_lib.energy_model import (
    EnergyModel,
    generate_energy_images,
    tensor_to_png_bytes as energy_tensor_to_png_bytes,
)
from helper_lib.gan_generator import generate_digit_base64
from helper_lib.model import get_model
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
import spacy

app = FastAPI(
    title="APAN 5560 Generative AI API",
    description=(
        "Text embeddings, CIFAR-10 classification, MNIST GAN generation, and "
        "CIFAR-10 Energy Model and Diffusion Model generation."
    ),
    version="1.1.0",
)
device = "cuda" if torch.cuda.is_available() else "cpu"
base_dir = Path(__file__).resolve().parent

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
checkpoint_path = base_dir / "checkpoints" / "cnn_cifar10.pth"

if checkpoint_path.exists():
    classifier_model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    classifier_model.to(device)
    classifier_model.eval()
else:
    classifier_model = None
nlp = spacy.load("en_core_web_lg")


def load_energy_checkpoint():
    checkpoint_path = base_dir / "checkpoints" / "cifar10_energy.pth"
    if not checkpoint_path.exists():
        return None, {}, f"Checkpoint not found: {checkpoint_path.name}"
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model = EnergyModel(**checkpoint.get("model_config", {}))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        return model, checkpoint.get("sampling_config", {}), None
    except Exception as exc:
        return None, {}, f"Checkpoint could not be loaded: {exc}"


def load_diffusion_checkpoint():
    checkpoint_path = base_dir / "checkpoints" / "cifar10_diffusion.pth"
    if not checkpoint_path.exists():
        return None, f"Checkpoint not found: {checkpoint_path.name}"
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model = CIFAR10Diffusion(
            **checkpoint.get("model_config", {}),
            **checkpoint.get("diffusion_config", {}),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        return model, None
    except Exception as exc:
        return None, f"Checkpoint could not be loaded: {exc}"


energy_model, energy_sampling_config, energy_model_error = load_energy_checkpoint()
diffusion_model, diffusion_model_error = load_diffusion_checkpoint()


class SimilarityRequest(BaseModel):
    text1: str
    text2: str


class EnergyGenerationRequest(BaseModel):
    num_images: int = Field(default=1, ge=1, le=4)
    steps: int = Field(default=100, ge=1, le=500)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)


class DiffusionGenerationRequest(BaseModel):
    num_images: int = Field(default=1, ge=1, le=4)
    steps: int = Field(default=100, ge=1, le=500)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)


@app.get("/")
def read_root():
    return {
        "message": "APAN 5560 Generative AI API is running.",
        "endpoints": {
            "embedding": "/embedding/{text}",
            "similarity": "/similarity",
            "predict_image": "/predict-image",
            "generate_digit": "/generate-digit",
            "generate_cifar10_energy": "/generate-cifar10/energy",
            "generate_cifar10_diffusion": "/generate-cifar10/diffusion",
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


@app.get("/generate-digit")
def generate_digit():
    """
    Generate one MNIST-style digit image using the saved GAN generator checkpoint.
    """

    try:
        return generate_digit_base64(device=device)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="GAN generator checkpoint not found. Run train_mnist_gan.py first.",
        ) from exc


@app.post("/generate-cifar10/energy")
def generate_cifar10_energy(request: EnergyGenerationRequest):
    """Generate RGB CIFAR-10-like images with Langevin Energy Model sampling."""

    if energy_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Energy Model is unavailable. Run train_cifar10_energy.py first. "
                f"{energy_model_error}"
            ),
        )

    step_size = float(energy_sampling_config.get("step_size", 0.1))
    noise_std = float(energy_sampling_config.get("noise_std", 0.01))
    generated_images = generate_energy_images(
        model=energy_model,
        num_images=request.num_images,
        device=device,
        steps=request.steps,
        step_size=step_size,
        noise_std=noise_std,
        seed=request.seed,
    )
    images = [
        {
            "image_base64": base64.b64encode(
                energy_tensor_to_png_bytes(image)
            ).decode("utf-8"),
            "image_format": "png",
        }
        for image in generated_images
    ]
    return {
        "model": "energy",
        "dataset": "CIFAR-10",
        "num_images": request.num_images,
        "steps": request.steps,
        "seed": request.seed,
        "device": device,
        "image_shape": [3, 32, 32],
        "normalization": "RGB channels normalized from [0, 1] to [-1, 1]",
        "images": images,
    }


@app.post("/generate-cifar10/diffusion")
def generate_cifar10_diffusion(request: DiffusionGenerationRequest):
    """Generate RGB CIFAR-10-like images with reverse diffusion sampling."""

    if diffusion_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Diffusion Model is unavailable. Run train_cifar10_diffusion.py "
                f"first. {diffusion_model_error}"
            ),
        )
    if request.steps > diffusion_model.timesteps:
        raise HTTPException(
            status_code=422,
            detail=(
                f"steps cannot exceed the checkpoint's "
                f"{diffusion_model.timesteps} diffusion timesteps."
            ),
        )

    generated_images = diffusion_model.sample(
        num_images=request.num_images,
        steps=request.steps,
        seed=request.seed,
    )
    images = [
        {
            "image_base64": base64.b64encode(
                diffusion_tensor_to_png_bytes(image)
            ).decode("utf-8"),
            "image_format": "png",
        }
        for image in generated_images
    ]
    return {
        "model": "diffusion",
        "dataset": "CIFAR-10",
        "num_images": request.num_images,
        "steps": request.steps,
        "training_timesteps": diffusion_model.timesteps,
        "seed": request.seed,
        "device": device,
        "image_shape": [3, 32, 32],
        "normalization": "RGB channels normalized from [0, 1] to [-1, 1]",
        "images": images,
    }
