import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from collections import OrderedDict


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class _RecursiveBlock(nn.Sequential):
    def __init__(self, n_feat=10):
        super(_RecursiveBlock, self).__init__()
        for d in range(n_feat):
            self.add_module('leaky_relu%d' % d,
                            nn.LeakyReLU(0.2, inplace=True))
            self.add_module('conv%d' % d,
                            nn.Conv2d(64, 64, kernel_size=3, padding=1))


class _FeatureEmbedding(nn.Module):
    def __init__(self, n_feat=10, n_recursive=1, local_residual='ns'):
        """Define feature embedding layer.
        Inputs:
        - n_feat: number of convolution layers in the recursive block.
        - n_recursive: number of times to recursively use the recursive block.
        - local_residual: local residual learning method to apply. Refer to
        paper for more details.
            + 'ns': no skip.
            + 'ds': distinct source.
            + 'ss': shared source.
        """
        super(_FeatureEmbedding, self).__init__()

        self.n_recursive = n_recursive
        self.local_residual = local_residual

        self.recursive_block = _RecursiveBlock(n_feat)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.upsample = nn.ConvTranspose2d(
            64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        if self.local_residual == 'ns':
            for i in range(self.n_recursive):
                x = self.recursive_block(x)
        elif self.local_residual == 'ds':
            for i in range(self.n_recursive):
                x_residual = self.recursive_block(x)
                x = x + x_residual
        else:
            source = x
            for i in range(self.n_recursive):
                x_residual = self.recursive_block(x)
                x = source + x_residual

        x = self.leaky_relu(x)
        x = self.upsample(x)

        return x


class LapSRN(nn.Module):
    def __init__(self, img_channels, upscale_factor, n_feat=10, n_recursive=1,
                 local_residual='ns'):
        """
        upscale_factor must be power of 2.
        """
        super(LapSRN, self).__init__()

        self.n_levels = int(np.log2(upscale_factor))
        self.local_residual = local_residual

        self.low_level_features = nn.Conv2d(
            img_channels, 64, kernel_size=3, padding=1)

        self.upsamplings = nn.ModuleList()
        self.embeddings = nn.ModuleList()
        self.residuals = nn.ModuleList()
        for i in range(self.n_levels):
            self.upsamplings.append(
                nn.ConvTranspose2d(img_channels, img_channels, kernel_size=4, 
                                   stride=2, padding=1, bias=False))
            self.embeddings.append(_FeatureEmbedding(n_feat, n_recursive, local_residual))
            self.residuals.append(nn.Conv2d(64, img_channels, kernel_size=3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, _ = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, h).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        features = self.low_level_features(x)

        # Result at each level.
        res = []

        identity = x
        for i in range(self.n_levels):
            identity = self.upsamplings[i](identity)
            features = self.embeddings[i](features)
            residual = self.residuals[i](features)
            res.append(identity + residual)

        return res[-1]