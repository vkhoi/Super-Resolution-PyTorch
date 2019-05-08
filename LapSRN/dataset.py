import numpy as np

from os import listdir
from os.path import join

from PIL import Image
from numpy import array
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomCrop, CenterCrop

from utilities import _rgb2ycbcr


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.JPEG'])


class SuperResDataset(Dataset):
    def __init__(self, image_dir, upscale_factor, img_channels, crop_size=-1,
                 augmentations=None, resampling=64):
        """
        - crop_size = -1 only when testing.
        """
        super(SuperResDataset, self).__init__()

        # Get image filenames.
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) \
                                if is_image_file(x)]
        # Read all images.
        self.images = []
        for image_filename in self.image_filenames:
            image = Image.open(image_filename).convert('RGB')
            if image.size[0] < crop_size or image.size[1] < crop_size:
                continue
            if img_channels == 1:
                image = _rgb2ycbcr(image)
                image, _, _ = image.split()
            self.images.append(image)

        self.len = len(self.images)

        self.upscale_factor = upscale_factor
        self.img_channels = img_channels
        self.crop_size = crop_size
        self.augmentations = augmentations
        self.resampling = resampling

    def __getitem__(self, index):
        input = self.images[index % self.len]

        if self.augmentations is not None:
            input = self.augmentations(input)

        if self.crop_size != -1:
            # For training, take random crop.
            if input.size[0] < self.crop_size or input.size[1] < self.crop_size:
                input = input.resize((self.crop_size, self.crop_size), Image.BICUBIC)
            else:
                input = RandomCrop(self.crop_size)(input)
        else:
            # For testing, we want to take the whole image.
            width, height = input.size[:2]
            width = width - (width % self.upscale_factor)
            height = height - (height % self.upscale_factor)
            input = CenterCrop((height, width))(input)

        # LapSRN employs deep supervision, so there will be multiple targets
        # (one target at each level).
        targets = [input.copy()]
        input = input.resize((input.size[0]//2, input.size[1]//2), Image.BICUBIC)

        n_levels = int(np.log2(self.upscale_factor))
        for i in range(n_levels - 1):
            targets = [input] + targets
            input = input.resize(
                (input.size[0]//2, input.size[1]//2), Image.BICUBIC)

        input = ToTensor()(input)
        for i in range(n_levels):
            targets[i] = ToTensor()(targets[i])

        return input, targets

    def __len__(self):
        return self.len * self.resampling