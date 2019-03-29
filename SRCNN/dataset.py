import numpy as np

from numpy import array
from os import listdir
from os.path import join
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomCrop, Resize, \
                                   CenterCrop, RandomHorizontalFlip
from PIL import Image

from utilities import rgb2ycrcb


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])


class SuperResDataset(Dataset):
    def __init__(self, image_dir, crop_size, upscale_factor, resampling=200,
                 transform=None):
        """
        The way the training data is sampled is as follows:
        - A batch of images are sampled from T91 dataset.
        - For each image, a random sub-image is cropped.
        - The sub-image is downsampled and upsampled so as to create training
        pairs.

        As we want to force one epoch to have around 200 sub-images per image in
        T91, we will sample each image multiple times (that's why a resampling
        argument is passed into this constructor).
        """
        super(SuperResDataset, self).__init__()

        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) \
                                if is_image_file(x)]
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.resampling = resampling
        self.len = len(self.image_filenames)
        self.transform = transform

    def __getitem__(self, index):
        # Open image and convert it to YCbCr.
        input = Image.open(
            self.image_filenames[index % self.len]).convert('RGB')
        input = rgb2ycrcb(input)

        # Only take the Y-channel.
        input, _, _ = input.split()
        if self.transform is not None:
            input = self.transform(input)
        target = input.copy()

        # Downsample then upsample.
        input = input.resize((input.size[0]//2, input.size[1]//2), 
                             Image.BICUBIC)
        input = input.resize((input.size[0]*2, input.size[1]*2),
                             Image.BICUBIC)

        input = ToTensor()(input)
        target = ToTensor()(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames) * self.resampling