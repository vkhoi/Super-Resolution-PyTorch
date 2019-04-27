import torch
import torch.nn as nn
import torch.nn.init as init


class FSRCNN(nn.Module):
    def __init__(self, img_channels, upscale_factor):
        """Fast Super-Resolution CNN.
        Inputs:
        - img_channels: either 1 (for Y-channel in YCbCr color space) or 3 (for RGB).
        - upscale_factor: upscale factor (2, 3, 4, 8).
        """
        super(FSRCNN, self).__init__()
        self.img_channels = img_channels
        self.upscale_factor = upscale_factor

        self.extraction = nn.Sequential(
            nn.Conv2d(self.img_channels, 56, kernel_size=5, padding=2),
            nn.PReLU())
        self.shrink = nn.Sequential(
            nn.Conv2d(56, 12, kernel_size=1),
            nn.PReLU())
        self.nonlinear_map = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(12, 56, kernel_size=1),
            nn.PReLU())
        self.deconv = nn.ConvTranspose2d(56, img_channels, kernel_size=9, 
                                         padding=4, stride=2, output_padding=1)

    def forward(self, x):
        x = self.extraction(x)
        x = self.shrink(x)
        x = self.nonlinear_map(x)
        x = self.expand(x)
        x = self.deconv(x)

        return x