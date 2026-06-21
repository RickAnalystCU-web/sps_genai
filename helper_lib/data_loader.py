from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_cifar10_loaders(
    data_dir: str = "data",
    batch_size: int = 64,
    num_workers: int = 2,
    validation_ratio: float = 0.10,
    seed: int = 42,
):
    """
    Create CIFAR10 train, validation, and test DataLoaders.

    CIFAR10 images are originally 32x32.
    Assignment 2 specifies a CNN input size of 64x64x3,
    so the transform resizes images to 64x64.

    The official CIFAR10 training set is split into:
    - 90% training
    - 10% validation

    The official CIFAR10 test set is kept separate and is not shuffled.
    """

    data_path = Path(data_dir)

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    full_train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transform,
    )

    val_size = int(len(full_train_dataset) * validation_ratio)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=4,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("First labels:", labels.tolist())
    print("Classes:", CIFAR10_CLASSES)