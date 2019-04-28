from torchvision.transforms import Compose, RandomHorizontalFlip

from augmentations import RandomResize, RandomRotate
from dataset import SuperResDataset


def augmentations():
    return Compose([
        RandomRotate(),
        RandomHorizontalFlip(p=0.5),
    ])


def get_training_set(img_dir, upscale_factor, crop_size):
    return SuperResDataset(img_dir, upscale_factor, crop_size,
                           augmentations=augmentations())


def get_val_set(img_dir, upscale_factor):
    return SuperResDataset(img_dir, upscale_factor, resampling=1)