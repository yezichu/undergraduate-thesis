from __future__ import print_function, division
from torch.utils.data import Dataset
import numpy as np
import os
import glob


class BratsDataset(Dataset):
    """Build a Dataset for brain tumor.

    Args:
        Dataset (dataset)
    -------------------------------------------------by YeziChu
    """
    def __init__(self, data_path, phase, transform=None):
        """Initialize the properties of the instance.

        Args:
            data_path (str): The path to store the image.
            phase (str): Train or valulation.
            transform (optional): Whether to perform Data augmentation techniques. Defaults to None.
        """
        self.data_path = data_path
        self.phase = phase
        self.transform = transform
        self.imgs_path = sorted(
            glob.glob(os.path.join(data_path, phase + 'Image/*')),
            key=lambda name: int(name[-10:-7] + name[-6:-4]))
        self.mask_path = sorted(
            glob.glob(os.path.join(data_path, phase + 'Mask/*')),
            key=lambda name: int(name[-10:-7] + name[-6:-4]))

    def __len__(self):
        """This is a function that gets the number of pictures in the path.

        Returns:
            int: Return the number of pictures in the path.
        """
        return len(self.imgs_path)

    def __getitem__(self, index):
        """This is a function that lets an object perform an iteration.

        Args:
            index (int): Defaults.

        Returns:
            array: Return image and mask.
        """
        image_path = self.imgs_path[index]
        mask_path = self.mask_path[index]
        image = self.load_img(image_path)
        image = image.transpose((3, 0, 1, 2))
        image = image.astype("float32")
        mask = self.load_img(mask_path)
        mask = mask.transpose((3, 0, 1, 2))
        mask = mask.astype("float32")
        img_and_mask = (image, mask)
        if self.transform:
            image, mask = self.transform(img_and_mask)

        return image, mask

    def load_img(self, file_path):
        """Load the image in the path.

        Args:
            file_path (str): The path to store the image.

        Returns:
            array: Return image matrix.
        """
        data = np.load(file_path)
        return data
