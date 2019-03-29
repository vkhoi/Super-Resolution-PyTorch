import argparse
import numpy as np
import torch

from PIL import Image
from torchvision.transforms import ToTensor

from model import SRCNN
from utilities import rgb2ycrcb, ycbcr2rgb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, 
                        help='input image to super resolve')
    parser.add_argument('--model', type=str, required=True,
                        help='model file')
    parser.add_argument('--output', type=str, required=True,
                        help='where to save the output image')
    args = parser.parse_args()

    img = Image.open(args.input).convert('RGB')
    img = rgb2ycrcb(img)
    y, cb, cr = img.split()

    ckpt = torch.load(args.model, map_location='cpu')
    model = SRCNN()
    model.load_state_dict(ckpt['model'])

    y = y.resize((y.size[0]*2, y.size[1]*2), Image.BICUBIC)
    input = ToTensor()(y).view(1, -1, y.size[1], y.size[0])

    out = model(input)
    out_img_y = out.detach().numpy().squeeze()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr])
    out_img = ycbcr2rgb(out_img)

    out_img.save(args.output)
    print('output image saved to', args.output)

