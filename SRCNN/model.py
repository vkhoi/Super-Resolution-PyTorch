import torch
import torch.nn as nn
import torch.nn.init as init


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x