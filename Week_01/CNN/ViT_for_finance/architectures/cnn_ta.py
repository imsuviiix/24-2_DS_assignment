import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=65, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 16 * 16, 32)  
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.interpolate(x, size=(64, 64))  
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x