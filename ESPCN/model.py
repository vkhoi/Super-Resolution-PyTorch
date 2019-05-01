import torch
import torch.nn as nn
import torch.nn.init as init


class ESPCN(nn.Module):
    def __init__(self, img_channels, upscale_factor):
        super(ESPCN, self).__init__()
        self.img_channels = img_channels
        self.upscale_factor = upscale_factor

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, img_channels*(upscale_factor**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x