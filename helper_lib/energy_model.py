"""Compact Energy-Based Model utilities for 32x32 RGB CIFAR-10 images."""

import io

import torch
import torch.nn as nn
from PIL import Image


class EnergyModel(nn.Module):
    """Map a normalized CIFAR-10 image to one scalar energy value."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.base_channels = base_channels

        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels * 4,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(
                base_channels * 4,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
        )
        self.energy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4 * 2 * 2, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return a tensor of shape ``(batch_size,)`` with one energy per image."""

        if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
            raise ValueError(
                "EnergyModel expects images with shape (batch_size, 3, 32, 32)."
            )
        return self.energy_head(self.features(images)).squeeze(-1)


def langevin_sample(
    model: nn.Module,
    initial_images: torch.Tensor,
    steps: int = 60,
    step_size: float = 0.1,
    noise_std: float = 0.01,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Move images toward lower-energy regions using Langevin dynamics.

    Model parameters are temporarily frozen so sampling does not allocate their
    gradients. Gradients with respect to the input images remain enabled, which
    is required both during training and API inference.
    """

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative")

    parameter_states = [parameter.requires_grad for parameter in model.parameters()]
    was_training = model.training
    images = initial_images.detach()

    try:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        for _ in range(steps):
            images.requires_grad_(True)
            total_energy = model(images).sum()
            (image_gradients,) = torch.autograd.grad(total_energy, images)

            with torch.no_grad():
                if noise_std:
                    noise = torch.randn(
                        images.shape,
                        generator=generator,
                        device=images.device,
                        dtype=images.dtype,
                    )
                    images.add_(noise, alpha=noise_std)
                images.add_(image_gradients, alpha=-step_size)
                images.clamp_(-1.0, 1.0)

            images = images.detach()
    finally:
        for parameter, requires_grad in zip(model.parameters(), parameter_states):
            parameter.requires_grad_(requires_grad)
        model.train(was_training)

    return images


def generate_energy_images(
    model: nn.Module,
    num_images: int,
    device: str | torch.device,
    steps: int = 100,
    step_size: float = 0.1,
    noise_std: float = 0.01,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate CIFAR-10-like samples starting from uniform random noise."""

    if num_images < 1:
        raise ValueError("num_images must be at least 1")
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()
    initial_images = torch.empty(num_images, 3, 32, 32, device=device)
    initial_images.uniform_(-1.0, 1.0, generator=generator)
    return langevin_sample(
        model=model,
        initial_images=initial_images,
        steps=steps,
        step_size=step_size,
        noise_std=noise_std,
        generator=generator,
    )


def tensor_to_rgb_image(image_tensor: torch.Tensor) -> Image.Image:
    """Convert one normalized ``(3, 32, 32)`` tensor to a Pillow RGB image."""

    if image_tensor.shape != (3, 32, 32):
        raise ValueError("Expected one image tensor with shape (3, 32, 32).")
    pixels = (
        ((image_tensor.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    return Image.fromarray(pixels, mode="RGB")


def tensor_to_png_bytes(image_tensor: torch.Tensor) -> bytes:
    """Encode one normalized CIFAR-10 tensor as RGB PNG bytes."""

    buffer = io.BytesIO()
    tensor_to_rgb_image(image_tensor).save(buffer, format="PNG")
    return buffer.getvalue()
