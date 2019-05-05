import numpy as np

from PIL import Image


class RandomResize(object):
    def __init__(self, scales=[0.6, 0.7, 0.8, 0.9, 1.0]):
        self.scales = scales

    def __call__(self, image):
        scale = np.random.choice(self.scales)
        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)

        image = image.resize((new_width, new_height), Image.BICUBIC)

        return image


class RandomRotate(object):
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, image):
        angle = np.random.choice(self.angles)
        if angle > 0:
            image = image.rotate(angle, expand=True)

        return image
