"""Train a compact Energy-Based Model on CIFAR-10."""

import argparse
import random
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from helper_lib.data_loader import get_cifar10_generative_loader
from helper_lib.energy_model import EnergyModel, langevin_sample


class ReplayBuffer:
    """Small CPU replay buffer for persistent contrastive divergence."""

    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self.images = torch.empty(0, 3, 32, 32)

    def sample(self, batch_size: int, device: str, random_fraction: float = 0.05):
        random_count = batch_size
        replay_images = torch.empty(0, 3, 32, 32)

        if len(self.images):
            replay_count = min(
                len(self.images),
                batch_size - max(1, int(batch_size * random_fraction)),
            )
            indices = torch.randint(len(self.images), (replay_count,))
            replay_images = self.images[indices]
            random_count = batch_size - replay_count

        random_images = torch.empty(random_count, 3, 32, 32).uniform_(-1.0, 1.0)
        initial_images = torch.cat((replay_images, random_images), dim=0)
        permutation = torch.randperm(batch_size)
        return initial_images[permutation].to(device, non_blocking=True)

    def update(self, images: torch.Tensor):
        new_images = images.detach().cpu()
        self.images = torch.cat((self.images, new_images), dim=0)
        if len(self.images) > self.capacity:
            self.images = self.images[-self.capacity :]


def resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return requested_device


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace):
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader = get_cifar10_generative_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
    )
    model = EnergyModel(base_channels=args.base_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.0, 0.999))
    replay_buffer = ReplayBuffer(capacity=args.replay_size)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Energy Model parameters: {parameter_count:,}")

    last_average_loss = float("nan")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        processed_batches = 0
        progress = tqdm(train_loader, desc=f"Energy epoch {epoch + 1}/{args.epochs}")

        for batch_index, (real_images, _) in enumerate(progress):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break

            real_images = real_images.to(device, non_blocking=True)
            negative_start = replay_buffer.sample(real_images.size(0), device=device)
            negative_images = langevin_sample(
                model=model,
                initial_images=negative_start,
                steps=args.langevin_steps,
                step_size=args.langevin_step_size,
                noise_std=args.langevin_noise,
            )

            optimizer.zero_grad(set_to_none=True)
            real_energy = model(real_images)
            negative_energy = model(negative_images.detach())
            contrastive_loss = real_energy.mean() - negative_energy.mean()
            energy_regularization = args.energy_regularization * (
                real_energy.square().mean() + negative_energy.square().mean()
            )
            loss = contrastive_loss + energy_regularization
            loss.backward()
            optimizer.step()

            replay_buffer.update(negative_images)
            running_loss += loss.item()
            processed_batches += 1
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                real=f"{real_energy.mean().item():.3f}",
                negative=f"{negative_energy.mean().item():.3f}",
            )

        if not processed_batches:
            raise RuntimeError("No training batches were processed.")
        last_average_loss = running_loss / processed_batches
        print(
            f"Epoch {epoch + 1}/{args.epochs} complete - "
            f"batches: {processed_batches}, average loss: {last_average_loss:.6f}"
        )

    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {"base_channels": args.base_channels},
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "sampling_config": {
            "step_size": args.langevin_step_size,
            "noise_std": args.langevin_noise,
            "recommended_steps": max(100, args.langevin_steps),
        },
        "training_config": vars(args) | {"checkpoint_path": str(args.checkpoint_path)},
        "epochs_completed": args.epochs,
        "average_loss": last_average_loss,
    }
    torch.save(checkpoint, args.checkpoint_path)
    print(f"Energy Model checkpoint saved to: {args.checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--langevin-steps", type=int, default=20)
    parser.add_argument("--langevin-step-size", type=float, default=0.1)
    parser.add_argument("--langevin-noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default="data/assignment4")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--replay-size", type=int, default=10_000)
    parser.add_argument("--energy-regularization", type=float, default=1e-3)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Limit batches per epoch for smoke tests; 0 uses the full dataset.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("checkpoints") / "cifar10_energy.pth",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
