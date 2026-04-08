import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, f, w, s, in_channels, dropout_rate=0.25):
        super(ConvBlock, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=f, kernel_size=w, stride=s, padding='same')
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm1d(f)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=dropout_rate)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)