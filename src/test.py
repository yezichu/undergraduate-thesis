from __future__ import print_function, division
import glob
import os
from model.H_Resunet import H_Resunet
import numpy as np
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import torch
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
nodel = H_Resunet(80).to('cuda')


class BratsDataset(Dataset):
    def __init__(self, data_path, phase, transform=None):
        self.data_path = data_path
        self.phase = phase
        self.transform = transform
        self.imgs_path = sorted(
            glob.glob(os.path.join(data_path, phase + 'Image/*')),
            key=lambda name: int(name[-10:-7] + name[-6:-4]))

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        image = self.load_img(image_path)
        image = image.transpose((3, 0, 1, 2))
        image = image.astype("float32")
        return image

    def load_img(self, file_path):
        data = np.load(file_path)
        return data


def GetPatchPosition(PatchPath):
    npName = os.path.basename(PatchPath)
    firstName = npName
    overNum = npName.find(".npy")
    npName = npName[0:overNum]
    PeopleName = npName
    overNum = npName.find("_")
    while (overNum != -1):
        npName = npName[overNum + 1:len(npName)]
        overNum = npName.find("_")
    overNum = firstName.find("_" + npName + ".npy")
    PeopleName = PeopleName[0:overNum]
    return int(npName), PeopleName


def get_dataloader(
    dataset,
    data_path: str,
    phase: str,
    transform=None,
    batch_size: int = 1,
    num_workers: int = 0,
):
    dataset = dataset(data_path, phase, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )
    return dataloader


mask_path = glob.glob("/home/mimt/jupyter/wy/testImage/*")
mask_path = sorted(mask_path, key=lambda name: int(name[-10:-7] + name[-6:-4]))
data_path = "/home/mimt/jupyter/wy/"
val_loader = get_dataloader(BratsDataset, data_path, phase='test')

nodel.load_state_dict(
    torch.load("/home/mimt/jupyter/wy/train/daima/96_epoch_model.pth"))
box = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2),
       (0, 2, 0), (0, 2, 1), (0, 2, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2),
       (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2),
       (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 1, 1), (2, 1, 2),
       (2, 2, 0), (2, 2, 1), (2, 2, 2)]
predicts = []
with torch.no_grad():
    for mynum, inputs in tqdm(enumerate(val_loader), total=len(val_loader)):
        image = inputs
        image = image.to(device)
        image, image1 = nodel(image)
        output = torch.sigmoid(image).data.cpu().numpy()
        predicts.append(output)
        namepath = mask_path[mynum]
        PatchPosition, NameNow = GetPatchPosition(namepath)
        LastName = NameNow
        if len(predicts) == 27:
            OneWT = np.zeros([180, 180, 180], dtype=np.uint8)
            OneTC = np.zeros([180, 180, 180], dtype=np.uint8)
            OneET = np.zeros([180, 180, 180], dtype=np.uint8)
            for i in range(27):
                image = predicts[i]
                image = image.squeeze(0)
                for idz in range(10, image.shape[1] - 10):
                    for idx in range(10, image.shape[2] - 10):
                        for idy in range(10, image.shape[3] - 10):
                            if image[0, idz, idx, idy] > 0.5:  # WT拼接
                                OneWT[box[i][0] * 60 + idz - 10,
                                      box[i][1] * 60 + idx - 10,
                                      box[i][2] * 60 + idy - 10] = 1
                            if image[1, idz, idx, idy] > 0.5:  # TC拼接
                                OneTC[box[i][0] * 60 + idz - 10,
                                      box[i][1] * 60 + idx - 10,
                                      box[i][2] * 60 + idy - 10] = 1
                            if image[2, idz, idx, idy] > 0.5:  # ET拼接
                                OneET[box[i][0] * 60 + idz - 10,
                                      box[i][1] * 60 + idx - 10,
                                      box[i][2] * 60 + idy - 10] = 1
            predicts.clear()
            mask1 = np.zeros([155, 240, 240], dtype=np.uint8)
            mask2 = np.zeros([155, 240, 240], dtype=np.uint8)
            mask3 = np.zeros([155, 240, 240], dtype=np.uint8)
            OnePeople = np.zeros([155, 240, 240], dtype=np.uint8)
            mask1[:, 30:210, 30:210] = OneWT[13:168, :, :]
            mask2[:, 30:210, 30:210] = OneTC[13:168, :, :]
            mask3[:, 30:210, 30:210] = OneET[13:168, :, :]
            for idz in range(mask1.shape[0]):
                for idx in range(mask1.shape[1]):
                    for idy in range(mask1.shape[2]):
                        if (mask1[idz, idx, idy] == 1):
                            OnePeople[idz, idx, idy] = 2
                        if (mask2[idz, idx, idy] == 1):
                            OnePeople[idz, idx, idy] = 1
                        if (mask3[idz, idx, idy] == 1):
                            OnePeople[idz, idx, idy] = 4
            saveout = sitk.GetImageFromArray(OnePeople)
            saveout.SetOrigin((-0.0, -239.0, 0.0))
            sitk.WriteImage(saveout, LastName + ".nii.gz")
