import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, num_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, 16, 3, padding='same')
        self.conv2 = nn.Conv2d(16, 16, 3,  padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 1, 3,  padding='same')
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.conv3(x)
        return x


class Decoder(nn.Module):

    def __init__(self, num_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding='same')
        self.conv2 = nn.Conv2d(16, 16, 3,  padding='same')
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(16, num_channels, 3,  padding='same')
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.upsample(x)
        x = self.conv3(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.encoder = Encoder(num_channels=num_channels)
        self.decoder = Decoder(num_channels=num_channels)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
