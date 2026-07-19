"""Train a compact noise-prediction diffusion model on CIFAR-10."""

import argparse
import random
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from helper_lib.data_loader import get_cifar10_generative_loader
from helper_lib.diffusion_model import CIFAR10Diffusion


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
    model = CIFAR10Diffusion(
        base_channels=args.base_channels,
        time_embedding_dim=args.time_embedding_dim,
        timesteps=args.diffusion_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Diffusion Model parameters: {parameter_count:,}")

    last_average_loss = float("nan")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        processed_batches = 0
        progress = tqdm(train_loader, desc=f"Diffusion epoch {epoch + 1}/{args.epochs}")

        for batch_index, (clean_images, _) in enumerate(progress):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break

            clean_images = clean_images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(clean_images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            processed_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

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
        "model_config": {
            "base_channels": args.base_channels,
            "time_embedding_dim": args.time_embedding_dim,
        },
        "diffusion_config": {
            "timesteps": args.diffusion_timesteps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
        },
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "training_config": vars(args) | {"checkpoint_path": str(args.checkpoint_path)},
        "epochs_completed": args.epochs,
        "average_loss": last_average_loss,
    }
    torch.save(checkpoint, args.checkpoint_path)
    print(f"Diffusion Model checkpoint saved to: {args.checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--diffusion-timesteps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default="data/assignment4")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--time-embedding-dim", type=int, default=64)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Limit batches per epoch for smoke tests; 0 uses the full dataset.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("checkpoints") / "cifar10_diffusion.pth",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
