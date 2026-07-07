import base64
import io
from pathlib import Path

import torch
from PIL import Image

from helper_lib.gan_model import Generator


DEFAULT_GAN_CHECKPOINT = Path("checkpoints") / "mnist_gan_generator.pth"


def generate_digit_base64(
    checkpoint_path: Path = DEFAULT_GAN_CHECKPOINT,
    device: str | None = None,
    noise_dim: int = 100,
):
    """
    Load the saved MNIST GAN generator and return one generated digit as PNG base64.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_path}")

    generator = Generator(noise_dim=noise_dim)
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.to(device)
    generator.eval()

    noise = torch.randn(1, noise_dim, device=device)

    with torch.no_grad():
        generated = generator(noise).squeeze(0).squeeze(0)
        generated = ((generated + 1.0) / 2.0).clamp(0.0, 1.0)
        image_array = (generated.cpu() * 255).byte().numpy()

    image = Image.fromarray(image_array, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "message": "Generated one MNIST-style digit image.",
        "image_base64": image_base64,
        "image_format": "png",
        "device": device,
    }
