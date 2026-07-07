import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator for MNIST digit images.

    Input: noise vector of shape (batch_size, 100)
    Output: image tensor of shape (batch_size, 1, 28, 28)
    """

    def __init__(self, noise_dim: int = 100):
        super().__init__()
        self.noise_dim = noise_dim

        self.fc = nn.Linear(noise_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 128, 7, 7)
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator for MNIST digit images.

    Input: image tensor of shape (batch_size, 1, 28, 28)
    Output: real/fake probability of shape (batch_size, 1)
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    generator = Generator()
    discriminator = Discriminator()

    noise = torch.randn(4, 100)
    fake_images = generator(noise)
    probabilities = discriminator(fake_images)

    print(generator)
    print(discriminator)
    print("Noise shape:", noise.shape)
    print("Generated image shape:", fake_images.shape)
    print("Discriminator output shape:", probabilities.shape)
