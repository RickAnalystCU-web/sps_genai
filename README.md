# APAN 5560 Generative AI FastAPI Project

This project implements several model inference endpoints using FastAPI. It includes spaCy text embeddings and similarity, CIFAR10 image classification with a CNN, and MNIST-style image generation with a GAN.

## Features / Endpoints

* `GET /`
  Returns a basic API status message and lists available endpoints.

* `GET /embedding/{text}`
  Generates spaCy vector information for the input text, including vector availability, dimension, norm, and a short embedding preview.

* `POST /similarity`
  Compares two input texts using spaCy word vectors and returns their semantic similarity score.

* `POST /predict-image`
  Accepts an uploaded image file and predicts its CIFAR10 class using the saved CNN classifier checkpoint.

* `GET /generate-digit`
  Loads the saved GAN generator checkpoint, samples random noise, generates one MNIST-style digit image, and returns it as PNG base64.

## Run Locally

Start the FastAPI server:

```powershell
.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

## Run with Docker

Build and start the app:

```bash
docker compose up --build
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

## Training Scripts

* `train_cifar10.py` trains the CNN classifier for CIFAR10 image classification.
* `train_mnist_gan.py` trains the GAN generator for MNIST-style digit generation.

Datasets are not committed to the repository. Saved model checkpoints are included so the API inference endpoints can run without retraining.

## Repository

https://github.com/RickAnalystCU-web/sps_genai
