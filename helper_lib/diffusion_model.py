"""Compact timestep-conditioned diffusion model for 32x32 RGB CIFAR-10."""

import io
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class SinusoidalTimeEmbedding(nn.Module):
    """Encode integer diffusion timesteps with sine and cosine frequencies."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        if embedding_dim < 4:
            raise ValueError("embedding_dim must be at least 4")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        scale = math.log(10_000) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half_dim, device=timesteps.device)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat((angles.sin(), angles.cos()), dim=1)
        if embeddings.shape[1] < self.embedding_dim:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class ResidualTimeBlock(nn.Module):
    """Residual convolution block conditioned on a timestep embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, images: torch.Tensor, time_embedding: torch.Tensor):
        hidden = self.conv1(F.silu(self.norm1(images)))
        hidden = hidden + self.time_projection(time_embedding)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(images)


class SmallUNet(nn.Module):
    """Small UNet-like network that predicts noise for CIFAR-10 images."""

    def __init__(self, base_channels: int = 32, time_embedding_dim: int = 64):
        super().__init__()
        if base_channels % 8:
            raise ValueError("base_channels must be divisible by 8")
        self.base_channels = base_channels
        self.time_embedding_dim = time_embedding_dim

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 2, time_embedding_dim),
        )
        self.input_conv = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)

        self.encoder1 = ResidualTimeBlock(
            base_channels, base_channels, time_embedding_dim
        )
        self.down1 = nn.Conv2d(
            base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.encoder2 = ResidualTimeBlock(
            base_channels * 2, base_channels * 2, time_embedding_dim
        )
        self.down2 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.middle1 = ResidualTimeBlock(
            base_channels * 4, base_channels * 4, time_embedding_dim
        )
        self.middle2 = ResidualTimeBlock(
            base_channels * 4, base_channels * 4, time_embedding_dim
        )

        self.up1 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.decoder1 = ResidualTimeBlock(
            base_channels * 4, base_channels * 2, time_embedding_dim
        )
        self.up2 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.decoder2 = ResidualTimeBlock(
            base_channels * 2, base_channels, time_embedding_dim
        )

        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_conv = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def forward(self, noisy_images: torch.Tensor, timesteps: torch.Tensor):
        if noisy_images.ndim != 4 or noisy_images.shape[1:] != (3, 32, 32):
            raise ValueError(
                "SmallUNet expects images with shape (batch_size, 3, 32, 32)."
            )
        time_embedding = self.time_embedding(timesteps)

        input_features = self.input_conv(noisy_images)
        skip1 = self.encoder1(input_features, time_embedding)
        skip2 = self.encoder2(self.down1(skip1), time_embedding)
        hidden = self.down2(skip2)
        hidden = self.middle1(hidden, time_embedding)
        hidden = self.middle2(hidden, time_embedding)

        hidden = self.up1(hidden)
        hidden = self.decoder1(torch.cat((hidden, skip2), dim=1), time_embedding)
        hidden = self.up2(hidden)
        hidden = self.decoder2(torch.cat((hidden, skip1), dim=1), time_embedding)
        return self.output_conv(F.silu(self.output_norm(hidden)))


class CIFAR10Diffusion(nn.Module):
    """Noise-prediction diffusion process with a compact CIFAR-10 UNet."""

    def __init__(
        self,
        base_channels: int = 32,
        time_embedding_dim: int = 64,
        timesteps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("Expected 0 < beta_start < beta_end < 1")

        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.network = SmallUNet(
            base_channels=base_channels,
            time_embedding_dim=time_embedding_dim,
        )

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, images: torch.Tensor):
        return values.gather(0, timesteps).view(-1, 1, 1, 1).to(images.dtype)

    def forward(self, noisy_images: torch.Tensor, timesteps: torch.Tensor):
        return self.network(noisy_images, timesteps)

    def q_sample(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the closed-form forward noising process q(x_t | x_0)."""

        if noise is None:
            noise = torch.randn_like(clean_images)
        signal = self._extract(self.sqrt_alpha_bars, timesteps, clean_images)
        noise_rate = self._extract(
            self.sqrt_one_minus_alpha_bars, timesteps, clean_images
        )
        return signal * clean_images + noise_rate * noise, noise

    def training_loss(self, clean_images: torch.Tensor) -> torch.Tensor:
        """Return MSE between sampled noise and the network's noise prediction."""

        timesteps = torch.randint(
            0,
            self.timesteps,
            (clean_images.size(0),),
            device=clean_images.device,
        )
        noisy_images, noise = self.q_sample(clean_images, timesteps)
        predicted_noise = self(noisy_images, timesteps)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        num_images: int,
        steps: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate images using a deterministic DDIM-style reverse process."""

        if steps is None:
            steps = self.timesteps
        if not 1 <= steps <= self.timesteps:
            raise ValueError(f"steps must be between 1 and {self.timesteps}")
        device = next(self.parameters()).device
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()
        images = torch.randn(
            num_images,
            3,
            32,
            32,
            device=device,
            generator=generator,
        )

        sample_indices = torch.linspace(
            self.timesteps - 1,
            0,
            steps,
            device=device,
        ).round().long()
        sample_indices = torch.unique_consecutive(sample_indices)

        was_training = self.training
        self.eval()
        try:
            for index, timestep_value in enumerate(sample_indices):
                timestep = torch.full(
                    (num_images,),
                    int(timestep_value.item()),
                    device=device,
                    dtype=torch.long,
                )
                predicted_noise = self(images, timestep)
                alpha_bar = self.alpha_bars[timestep_value]
                predicted_clean = (
                    images - (1.0 - alpha_bar).sqrt() * predicted_noise
                ) / alpha_bar.sqrt()
                predicted_clean.clamp_(-1.0, 1.0)

                if index + 1 == len(sample_indices):
                    images = predicted_clean
                else:
                    next_alpha_bar = self.alpha_bars[sample_indices[index + 1]]
                    images = (
                        next_alpha_bar.sqrt() * predicted_clean
                        + (1.0 - next_alpha_bar).sqrt() * predicted_noise
                    )
        finally:
            self.train(was_training)

        return images.clamp(-1.0, 1.0)


def tensor_to_rgb_image(image_tensor: torch.Tensor) -> Image.Image:
    """Convert one diffusion sample in [-1, 1] to a Pillow RGB image."""

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
    """Encode one normalized diffusion sample as RGB PNG bytes."""

    buffer = io.BytesIO()
    tensor_to_rgb_image(image_tensor).save(buffer, format="PNG")
    return buffer.getvalue()
