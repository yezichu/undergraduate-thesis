from typing import Tuple
import numpy as np
import torch


class RandomMirrorFlip(object):
    # Random flip transformation.

    def __init__(self, p=0.5):
        """Initialize the properties of the instance.

        Args:
            p (float, optional): The probability. Defaults to 0.5.
        """
        super().__init__()
        self.p = p

    def __call__(self,
                 img_and_mask) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Overloaded operator.

        Args:
            img_and_mask ([type]): Image and mask tuples.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Return the shifted image and mask.
        """
        image, mask = img_and_mask
        if torch.rand(1) < self.p:
            image = np.flip(image, axis=[1, 2, 3])
            mask = np.flip(mask, axis=[1, 2, 3])

        return image.copy(), mask.copy()


class RandomRotation90(object):
    # Random rotation transformation.

    def __init__(self, p=0.5, num_rot=(1, 2, 3), axes=(0, 1, 2)):
        """Initialize the properties of the instance.

        Args:
            p (float, optional): The probability. Defaults to 0.5.
            num_rot (tuple, optional): The axis of the original data. Defaults to (1, 2, 3).
            axes (tuple, optional): The axis of the data after flipping. Defaults to (0, 1, 2).
        """
        super().__init__()
        self.p = p
        self.num_rot = num_rot
        self.axes = axes

    def __call__(self,
                 img_and_mask) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Overloaded operator.

        Args:
            img_and_mask ([type]): Image and mask tuples.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Return the shifted image and mask.
        """
        image, mask = img_and_mask
        if torch.rand(1) < self.p:
            num_rot1 = np.random.choice(self.num_rot)
            axes = np.random.choice(self.axes, size=2, replace=False)
            axes.sort()

            axes_data = [i + 1 for i in axes]
            image = np.rot90(image, num_rot1, axes_data)
            mask = np.rot90(mask, num_rot1, axes_data)
        return image.copy(), mask.copy()
        