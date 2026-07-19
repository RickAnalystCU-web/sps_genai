# APAN 5560 Generative AI FastAPI Project

This project implements several model inference endpoints using FastAPI. It includes spaCy text embeddings and similarity, CIFAR10 image classification with a CNN, MNIST-style image generation with a GAN, and CIFAR-10 image generation with Energy and Diffusion models.

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

* `POST /generate-cifar10/energy`
  Generates one to four 32x32 RGB images using Langevin sampling from a trained CIFAR-10 Energy Model.

* `POST /generate-cifar10/diffusion`
  Generates one to four 32x32 RGB images using a trained CIFAR-10 Diffusion Model.

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
* `train_cifar10_energy.py` trains the CIFAR-10 Energy Model.
* `train_cifar10_diffusion.py` trains the CIFAR-10 Diffusion Model.

Datasets are not committed to the repository. Saved model checkpoints are included so the API inference endpoints can run without retraining.

## Assignment 4: CIFAR-10 Energy and Diffusion Models

Both Assignment 4 models use native 32x32 RGB CIFAR-10 images. Pixels are normalized per channel with mean `0.5` and standard deviation `0.5`, mapping `[0, 1]` input values to approximately `[-1, 1]`.

The training scripts store/download their separate dataset copy under the ignored `data/assignment4` directory by default, leaving the existing Assignment 2 classifier data unchanged.

### Energy Model

The Energy Model is a compact convolutional network that produces one scalar energy per image. Training lowers the energy of real CIFAR-10 images and raises the energy of negative samples generated with replay-buffer Langevin dynamics.

The completed Windows/CUDA run trained for 10 epochs with batch size 128 and 20 Langevin steps. Windows multiprocessing was disabled because worker processes produced `WinError 5` on the training machine:

```powershell
.venv\Scripts\python.exe train_cifar10_energy.py --device cuda --epochs 10 --batch-size 128 --langevin-steps 20 --num-workers 0
```

The trained checkpoint is saved to `checkpoints/cifar10_energy.pth`. A short smoke-test command is:

```powershell
.venv\Scripts\python.exe train_cifar10_energy.py --device cuda --epochs 1 --batch-size 8 --langevin-steps 2 --max-train-batches 1 --num-workers 0 --checkpoint-path checkpoints/energy_smoke.pth
```

### Diffusion Model

The Diffusion Model uses a small timestep-conditioned UNet to predict noise. Its linear beta schedule defaults to 200 training timesteps, and API generation supports fewer deterministic reverse steps for lower latency.

The completed Windows/CUDA run trained for 30 epochs with batch size 128 and a 200-timestep schedule:

```powershell
.venv\Scripts\python.exe train_cifar10_diffusion.py --device cuda --epochs 30 --batch-size 128 --diffusion-timesteps 200 --num-workers 0
```

The trained checkpoint is saved to `checkpoints/cifar10_diffusion.pth`. A short smoke-test command is:

```powershell
.venv\Scripts\python.exe train_cifar10_diffusion.py --device cuda --epochs 1 --batch-size 8 --diffusion-timesteps 20 --max-train-batches 1 --num-workers 0 --checkpoint-path checkpoints/diffusion_smoke.pth
```

### API Requests and Responses

Energy Model example:

```json
{
  "num_images": 1,
  "steps": 100,
  "seed": 42
}
```

Send it to `POST /generate-cifar10/energy`.

Diffusion Model example:

```json
{
  "num_images": 1,
  "steps": 100,
  "seed": 42
}
```

Send it to `POST /generate-cifar10/diffusion`. Diffusion `steps` cannot exceed the number of timesteps stored in the checkpoint. Both endpoints allow 1-4 images and return metadata plus an `images` list containing base64-encoded RGB PNG data:

```json
{
  "model": "energy or diffusion",
  "dataset": "CIFAR-10",
  "image_shape": [3, 32, 32],
  "images": [
    {
      "image_base64": "...",
      "image_format": "png"
    }
  ]
}
```

CUDA is strongly recommended for training and multi-step generation. Both models can run on CPU, but Energy Model Langevin sampling and Diffusion reverse sampling will be substantially slower. If a trained checkpoint is missing or incompatible, only its corresponding endpoint returns HTTP 503; the rest of the API continues to run.

The included sample grids verify the complete generation and PNG-encoding pipeline. With these compact assignment-scale models, generated images remain abstract and noisy rather than consistently showing clearly recognizable CIFAR-10 objects; no production-quality claim is intended.

## Repository

https://github.com/RickAnalystCU-web/sps_genai
