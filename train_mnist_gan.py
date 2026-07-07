import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from helper_lib.gan_model import Discriminator, Generator


def get_mnist_loader(data_dir: str, batch_size: int, num_workers: int):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    dataset = datasets.MNIST(
        root=Path(data_dir),
        train=True,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )


def train_gan(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    noise_dim: int,
    data_dir: str,
    checkpoint_path: Path,
    num_workers: int,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader = get_mnist_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    generator = Generator(noise_dim=noise_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999),
    )

    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        total_d_loss = 0.0
        total_g_loss = 0.0

        for real_images, _ in progress_bar:
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)

            real_labels = torch.ones(current_batch_size, 1, device=device)
            fake_labels = torch.zeros(current_batch_size, 1, device=device)

            optimizer_d.zero_grad()

            real_outputs = discriminator(real_images)
            d_loss_real = criterion(real_outputs, real_labels)

            noise = torch.randn(current_batch_size, noise_dim, device=device)
            fake_images = generator(noise)
            fake_outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()

            noise = torch.randn(current_batch_size, noise_dim, device=device)
            fake_images = generator(noise)
            fake_outputs = discriminator(fake_images)
            g_loss = criterion(fake_outputs, real_labels)

            g_loss.backward()
            optimizer_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            progress_bar.set_postfix(
                d_loss=f"{d_loss.item():.4f}",
                g_loss=f"{g_loss.item():.4f}",
            )

        avg_d_loss = total_d_loss / len(train_loader)
        avg_g_loss = total_g_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}/{epochs} complete - "
            f"D loss: {avg_d_loss:.4f}, G loss: {avg_g_loss:.4f}"
        )

    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save(generator.state_dict(), checkpoint_path)
    print(f"Generator saved to: {checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple GAN on MNIST.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--noise-dim", type=int, default=100)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("checkpoints") / "mnist_gan_generator.pth",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_gan(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        noise_dim=args.noise_dim,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
