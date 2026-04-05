import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        # First Convolutional Block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1024, kernel_size=512, stride=4, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()

        # Second Convolutional Block
        self.conv2 = nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=64, stride=1, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        # Third Convolutional Block
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=64, stride=1, padding='same')
        self.pool3 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        # Fourth Convolutional Block
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=64, stride=1, padding='same')
        self.pool4 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()

        # Fifth Convolutional Block
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=64, stride=1, padding='same')
        self.pool5 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()

        # Sixth Convoluational Block
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=64, stride=1, padding='same')
        self.pool6 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.relu6 = nn.ReLU()

        # Fully Connected Layer
        fc = nn.Linear(in_features=2048, out_features=360)