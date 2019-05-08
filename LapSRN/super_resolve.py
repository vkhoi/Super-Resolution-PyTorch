import argparse
import numpy as np
import torch

from PIL import Image
from torchvision.transforms import ToTensor

from model import LapSRN
from utilities import _rgb2ycbcr, _ycbcr2rgb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='model file')
    parser.add_argument('--upscale_factor', type=int, required=True,
                        help='upscaling factor')
    parser.add_argument('--img_channels', type=int, required=True,
                        help='# of image channels (1 for Y, 3 for RGB)')
    parser.add_argument('--input', type=str, required=True, 
                        help='input image to super resolve')
    parser.add_argument('--output', type=str, required=True,
                        help='where to save the output image')
    args = parser.parse_args()

    img = Image.open(args.input).convert('RGB')
    if args.img_channels == 1:
        img = _rgb2ycbcr(img)
        img, cb, cr = img.split()

    ckpt = torch.load(args.model, map_location='cpu')
    model = LapSRN(img_channels=args.img_channels,
                   upscale_factor=args.upscale_factor)
    model.load_state_dict(ckpt['model'])

    input = ToTensor()(img).view(1, -1, img.size[1], img.size[0])

    out = model(input)[-1]
    out_img = out.detach().numpy().squeeze().transpose(1, 2, 0)
    out_img *= 255.0
    out_img = out_img.clip(0, 255)
    print(out_img.shape)
    if args.img_channels == 1:
        out_img = Image.fromarray(np.uint8(out_img), mode='L')
        out_img_cb = cb.resize(out_img.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img.size, Image.BICUBIC)
        out_img = Image.merge('YCbCr', [out_img, out_img_cb, out_img_cr])
        out_img = _ycbcr2rgb(out_img)
    else:
        out_img = Image.fromarray(np.uint8(out_img), mode='RGB')

    out_img.save(args.output)
    print('output image saved to', args.output)

