import torch
import torch.nn as nn


class AssignmentCNN(nn.Module):
    """
    CNN architecture for Assignment 2.

    Input: RGB image of size 64 x 64 x 3
    Conv2D: 16 filters, 3x3, stride 1, padding 1
    ReLU
    MaxPool2D: 2x2, stride 2
    Conv2D: 32 filters, 3x3, stride 1, padding 1
    ReLU
    MaxPool2D: 2x2, stride 2
    Flatten
    Fully connected: 100 units
    ReLU
    Fully connected: 10 output classes
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(model_name: str = "CNN", num_classes: int = 10):
    """
    Return a model by name. This follows the helper library style from the class activity.
    """

    model_name = model_name.upper()

    if model_name == "CNN":
        return AssignmentCNN(num_classes=num_classes)

    raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    model = get_model("CNN")
    dummy_input = torch.randn(4, 3, 64, 64)
    output = model(dummy_input)

    print(model)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)