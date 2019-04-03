from config import config
from dataset import SuperResDataset

from torchvision.transforms import Compose, \
                                   RandomHorizontalFlip, RandomVerticalFlip


def input_transform():
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
    ])


def get_training_set(upscale_factor):
    crop_size = config['crop_size'] - (config['crop_size'] % upscale_factor)

    return SuperResDataset(image_dir=config['TRAIN_DIR'],
                           upscale_factor=upscale_factor,
                           crop_size=crop_size, transform=input_transform())


def get_val_set(upscale_factor):
    # When doing validation, we only want to sample each image once, so
    # resampling = 1.
    return SuperResDataset(image_dir=config['VAL_DIR'],
                           upscale_factor=upscale_factor, resampling=1)