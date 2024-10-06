import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=4, stride=1, padding=1)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(3, 2, kernel_size=5, stride=1, padding=1)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(480, hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = self.fc1(x.reshape(x.shape[0], -1))
        # print(x.shape)
        x = self.tanh(x)

        return x


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            2, 3, kernel_size=5, stride=2, padding=1, output_padding=1
        )
        self.conv2 = nn.ConvTranspose2d(
            3, 1, kernel_size=[4, 5], stride=2, padding=1, output_padding=1
        )
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, 480)

    def forward(self, x):
        x = self.relu(self.fc1(x))

        x = x.reshape(-1, 2, 15, 16)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self, hidden_size):
        super(ConvAutoencoder, self).__init__()
        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder(hidden_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path)
