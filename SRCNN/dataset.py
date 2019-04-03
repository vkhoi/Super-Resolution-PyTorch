import numpy as np

from numpy import array
from scipy.ndimage import gaussian_filter
from os import listdir
from os.path import join
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomCrop, Resize, \
                                   CenterCrop, RandomHorizontalFlip
from PIL import Image

from config import config
from utilities import rgb2ycrcb


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])


class SuperResDataset(Dataset):
    def __init__(self, image_dir, upscale_factor, crop_size=-1, resampling=256,
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

        Inputs:
        - image_dir: directory to image set.
        - upscale_factor.
        - crop_size: -1 if take whole image size (for testing).
        - resampling: how many sub-images we want to sample per image.
        - transform: any custom transform we want to apply on the sampled image.
        """
        super(SuperResDataset, self).__init__()

        # Get filenames.
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) \
                                if is_image_file(x)]
        # Read all images.
        self.images = []
        for image_filename in self.image_filenames:
            image = Image.open(image_filename).convert('RGB')
            image = rgb2ycrcb(image)
            image, _, _ = image.split()
            self.images.append(image)

        self.upscale_factor = upscale_factor
        self.crop_size = crop_size
        self.resampling = resampling
        self.len = len(self.images)
        self.transform = transform

    def __getitem__(self, index):
        # Open image and convert it to YCbCr.
        input = self.images[index % self.len]

        if self.crop_size != -1:
            # For training, take random crop.
            input = RandomCrop(self.crop_size)(input)
        else:
            # For testing, we want to take the whole image.
            width, height = input.size[:2]
            width = width - (width % self.upscale_factor)
            height = height - (height % self.upscale_factor)
            input = CenterCrop((height, width))(input)

        # Apply custom transform.
        if self.transform is not None:
            input = self.transform(input)

        # Make a copy of it when it's still at high-res.
        target = input.copy()

        # Downsample then upsample to create image at low-res.
        input = input.resize(
            (input.size[0]//self.upscale_factor, input.size[1]//self.upscale_factor), 
            Image.BICUBIC)
        input = input.resize(
            (input.size[0]*self.upscale_factor, input.size[1]*self.upscale_factor),
            Image.BICUBIC)

        input = ToTensor()(input)
        target = ToTensor()(target)

        return input, target

    def __len__(self):
        return self.len * self.resampling