from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip

from dataset import SuperResDataset

TRAIN_DIR = '../../datasets/super-resolution/T91'
VAL_DIR = '../../datasets/super-resolution/Set5'


def input_transform(crop_size, upscale_factor):
    return Compose([
        RandomCrop(crop_size),
        RandomHorizontalFlip(p=0.5),
    ])


def get_training_set(upscale_factor, train_all=False):
    crop_size = 32 - (32 % upscale_factor)

    return SuperResDataset(image_dir=TRAIN_DIR, crop_size=crop_size,
                           upscale_factor=upscale_factor,
                           transform=input_transform(crop_size, upscale_factor))


def get_val_set(upscale_factor):
    crop_size = 32 - (32 % upscale_factor)

    # When doing validation, we only want to sample each image once, so
    # resampling = 1.
    return SuperResDataset(image_dir=VAL_DIR, crop_size=crop_size,
                           upscale_factor=upscale_factor, resampling=1)