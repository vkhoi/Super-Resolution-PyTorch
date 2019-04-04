import argparse
import numpy as np
import os
import torch

from numpy import array
from scipy.io import savemat
from torchvision.transforms import ToTensor
from PIL import Image

from model import SRCNN
from utilities import rgb2ycrcb, ycbcr2rgb, PSNR


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, 
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

    filenames = os.listdir(args.image_dir)
    image_filenames = [os.path.join(args.image_dir, x) for x in filenames \
                       if is_image_file(x)]
    image_filenames = sorted(image_filenames)

    model = SRCNN().to(device)
    if args.cuda:
        ckpt = torch.load(args.model)
    else:
        ckpt = torch.load(args.model, map_location='cpu')
    model.load_state_dict(ckpt['model'])

    # avg_bicubic_psnr = 0
    # avg_deep_psnr = 0

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
        img = rgb2ycrcb(img)
        img_y = array(img.split()[0], dtype=np.float32) / 255.0
        img_y = Image.fromarray(img_y, mode='F')

        # Downsample to get low-res image.
        lr_img_y = img_y.resize(
            (width//args.upscale_factor, height//args.upscale_factor),
            Image.BICUBIC)

        # Achive high-res using deep neural net.
        y = out_img_bicubic_y.copy()
        y = ToTensor()(y).view(1, -1, y.size[1], y.size[0])
        out_img_deep_y = model(y)[0].detach().numpy().squeeze()
        out_img_deep_y = out_img_deep_y.clip(0, 1)

        f = f.split('/')
        f = f[len(f) - 1]
        f = f.split('.')
        f = f[0]
        res[f] = out_img_deep_y

        # out_img_deep_y = Image.fromarray(out_img_deep_y, mode='F')

        # bicubic_psnr = PSNR(array(out_img_bicubic_y), array(img_y),
        #                     ignore_border=4)
        # deep_psnr = PSNR(array(out_img_deep_y), array(img_y), ignore_border=8)

        # avg_bicubic_psnr += bicubic_psnr
        # avg_deep_psnr += deep_psnr

        # print(f)
        # print('PSNR-Bicubic: {:.4f}'.format(bicubic_psnr))
        # print('PSNR-SRCNN: {:.4f}'.format(deep_psnr))
        # print('')

    # avg_bicubic_psnr /= len(image_filenames)
    # avg_deep_psnr /= len(image_filenames)

    # print('Average PSNR-Bicubic: {:.4f}'.format(avg_bicubic_psnr))
    # print('Average PSNR-SRCNN: {:.4f}'.format(avg_deep_psnr))

    savemat(args.output, res)
