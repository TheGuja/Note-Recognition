import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvBlock import ConvBlock

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        # Convolutional Blocks
        self.conv1 = ConvBlock(f=1024, w=512, s=4, in_channels=1)
        self.conv2 = ConvBlock(f=128, w=64, s=1, in_channels=1024)
        self.conv3 = ConvBlock(f=128, w=64, s=1, in_channels=128)
        self.conv4 = ConvBlock(f=128, w=64, s=1, in_channels=128)
        self.conv5 = ConvBlock(f=256, w=64, s=1, in_channels=128)
        self.conv6 = ConvBlock(f=512, w=64, s=1, in_channels=256)

        # Flatten the output
        self.flatten = nn.Flatten()

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=2048, out_features=360)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x