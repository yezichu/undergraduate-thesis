from typing import Tuple
import numpy as np
import random
import torch


class RandomIntensityScale(object):
    def __init__(self, min: float = 0.9, max: float = 1.1, p=0.5):
        super().__init__()
        self.min = min
        self.max = max
        self.p = p

    def __call__(self,
                 img_and_mask) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        image, mask = img_and_mask

        if torch.rand(1) < self.p:
            scale = random.uniform(self.min, self.max)
            image = image * scale

        return image.copy(), mask.copy()


class RandomIntensityShift(object):
    def __init__(self, min: float = -0.1, max: float = 0.1, p=0.5):
        super().__init__()
        self.min = min
        self.max = max
        self.p = p

    def __call__(self,
                 img_and_mask) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        image, mask = img_and_mask

        if torch.rand(1) < self.p:

            for i, modality in enumerate(image):

                shift = random.uniform(self.min, self.max)
                std = np.std(modality)
                image[i, ...] = modality + std * shift

        return image.copy(), mask.copy()


class RandomGaussianNoise(object):
    def __init__(self, p=0.5, noise_variance=(0, 0.5)):
        super().__init__()
        self.p = p
        self.noise_variance = noise_variance

    def __call__(self,
                 img_and_mask) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        image, mask = img_and_mask
        image = image.copy()
        mask = mask.copy()
        noised_image = image

        if torch.rand(1) < self.p:
            if self.noise_variance[0] == self.noise_variance[1]:
                variance = self.noise_variance[0]
            else:
                variance = random.uniform(self.noise_variance[0],
                                          self.noise_variance[1])

            noised_image = image + np.random.normal(
                0.0, variance, size=image.shape)

        return noised_image.copy(), mask.copy()
