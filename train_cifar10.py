from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_cifar10_loaders
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.trainer import train_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir="data",
        batch_size=64,
        num_workers=2,
        validation_ratio=0.10,
        seed=42,
    )

    model = get_model("CNN", num_classes=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=3,
    )

    print("Evaluating on validation set...")
    val_loss, val_accuracy = evaluate_model(
        model=model,
        data_loader=val_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    print("Evaluating on test set...")
    test_loss, test_accuracy = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / "cnn_cifar10.pth"
    torch.save(model.state_dict(), checkpoint_path)

    print(f"Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()