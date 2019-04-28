import argparse
import numpy as np
import os
import torch

from numpy import array
from scipy.io import savemat
from torchvision.transforms import ToTensor
from PIL import Image

from model import SRCNN
from utilities import _rgb2ycbcr, _ycbcr2rgb


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True, 
                        help='directory to image set for evaluating PSNR')
    parser.add_argument('--model', type=str, required=True,
                        help='model file')
    parser.add_argument('--upscale_factor', type=int, required=True,
                        help='upscale factor')
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

    model = SRCNN().to(device)
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
        img = _rgb2ycbcr(img)
        img_y = array(img.split()[0], dtype=np.float32) / 255.0
        img_y = Image.fromarray(img_y, mode='F')

        # Downsample to get low-res image.
        img_y = img_y.resize(
            (width//args.upscale_factor, height//args.upscale_factor),
            Image.BICUBIC)


        # Achive high-res using deep neural net.
        img_y = img_y.resize((width, height), Image.BICUBIC)
        img_y = ToTensor()(img_y).view(1, -1, img_y.size[1], img_y.size[0])
        sr_y = model(img_y)[0].detach().numpy().squeeze()
        sr_y = sr_y.clip(0, 1)

        # Get image filename.
        f = f.split('/')
        f = f[len(f) - 1]
        f = f.split('.')

        # Because MATLAB does not accept fieldname of struct to contain only
        # numbers, and there might be chance where the image name contains only
        # numbers, so we prepend a character to it.
        f = 'a' + f[0]
        # Save result to dict.
        res[f] = sr_y

    savemat(args.output, res)
