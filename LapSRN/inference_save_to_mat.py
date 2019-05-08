import argparse
import numpy as np
import os
import torch

from numpy import array
from scipy.io import savemat
from torchvision.transforms import ToTensor
from PIL import Image

from model import LapSRN
from utilities import _rgb2ycbcr


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True, 
                        help='directory to image set')
    parser.add_argument('--model', type=str, required=True,
                        help='model file')
    parser.add_argument('--upscale_factor', type=int, required=True,
                        help='upscale factor')
    parser.add_argument('--img_channels', type=int, required=True,
                        help='# of image channels (1 for Y, 3 for RGB)')
    parser.add_argument('--output', type=str, required=True,
                        help='output *.mat file')
    parser.add_argument('--cuda', action='store_true',
                        help='whether to use cuda')
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found')
    device = torch.device('cuda' if args.cuda else 'cpu')
    print('Use device:', device)

    filenames = os.listdir(args.img_dir)
    image_filenames = [os.path.join(args.img_dir, x) for x in filenames \
                       if is_image_file(x)]
    image_filenames = sorted(image_filenames)

    model = LapSRN(img_channels=1,
                   upscale_factor=args.upscale_factor,
                   n_feat=10,
                   n_recursive=1,
                   local_residual='ns').to(device)
    if args.cuda:
        ckpt = torch.load(args.model)
    else:
        ckpt = torch.load(args.model, map_location='cpu')
    model.load_state_dict(ckpt['model'])

    res = {}

    for i, f in enumerate(image_filenames):
        # Read test image.
        img = Image.open(f).convert('RGB')
        width, height = img.size[0], img.size[1]

        # Crop test image so that it has size that can be downsampled by the upscale factor.
        pad_width = width % args.upscale_factor
        pad_height = height % args.upscale_factor
        width -= pad_width
        height -= pad_height
        img = img.crop((0, 0, width, height))

        if args.img_channels == 1:
            img = _rgb2ycbcr(img)
            img_y = array(img.split()[0], dtype=np.float32) / 255.0
            img_y = Image.fromarray(img_y, mode='F')

        # Downsample to get low-res image.
        lr_img = img_y.resize(
            (width//args.upscale_factor, height//args.upscale_factor),
            Image.BICUBIC)

        # Achieve high-res using FSRCNN.
        y = lr_img.copy()
        y = ToTensor()(y).view(1, -1, y.size[1], y.size[0])
        out_img_deep_y = model(y)[-1].detach().numpy().squeeze()
        out_img_deep_y = out_img_deep_y.clip(0, 1)

        if args.img_channels == 3:
            # Must convert to luminance so that it can be evaluated by the
            # MATLAB code.
            out_img_deep_y = (out_img_deep_y * 255).astype(np.uint8)
            out_img_deep_y = _rgb2ycbcr(out_img_deep_y)

            # MATLAB code expects output as float, so we convert it again.
            out_img_deep_y = out_img_deep_y.astype(np.float32) / 255.

        # Get image filename.
        f = f.split('/')
        f = f[len(f) - 1]
        f = f.split('.')

        # Because MATLAB does not accept fieldname of struct to contain only
        # numbers, and there might be chance where the image name contains only
        # numbers, so we prepend a character to it.
        f = 'a' + f[0]
        # Save result to dict.
        res[f] = out_img_deep_y

    # Save mat file.
    savemat(args.output, res)
