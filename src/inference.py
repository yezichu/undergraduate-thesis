import glob
import os
from model.H_Resunet import H_Resunet
import numpy as np
from torch.utils.data import DataLoader
import SimpleITK as sitk
from model.BratsDataset import BratsDataset
import torch
from tqdm import tqdm
import pandas as pd
from metric.evaluation import dice_coef_metric_per_classes, jaccard_coef_metric_per_classes, sensitivity_metric_per_classes, specificity_metric_per_classes, hausdorff_95_per_classes

device = 'cuda' if torch.cuda.is_available() else 'cpu'
nodel = H_Resunet(80).to('cuda')


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


mask_path = glob.glob("/home/mimt/jupyter/wy/train/validationMask/*")
mask_path = sorted(mask_path, key=lambda name: int(name[-10:-7] + name[-6:-4]))
data_path = "/home/mimt/jupyter/wy/train/"
val_loader = get_dataloader(BratsDataset, data_path, phase='validation')

nodel.load_state_dict(
    torch.load("/home/mimt/jupyter/wy/train/daima/96_epoch_model.pth"))
box = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2),
       (0, 2, 0), (0, 2, 1), (0, 2, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2),
       (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 2, 0), (1, 2, 1), (1, 2, 2),
       (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 1, 1), (2, 1, 2),
       (2, 2, 0), (2, 2, 1), (2, 2, 2)]
predicts = []
masks = []
classes = ['WT', 'TC', 'ET']
savedir = '/home/mimt/jupyter/wy/train/baocun'
dice_scores_per_classes = {key: list() for key in classes}
iou_scores_per_classes = {key: list() for key in classes}
sens_scores_per_classes = {key: list() for key in classes}
spec_scores_per_classes = {key: list() for key in classes}
haus_scores_per_classes = {key: list() for key in classes}
with torch.no_grad():
    for mynum, inputs in tqdm(enumerate(val_loader), total=len(val_loader)):
        image, mask = inputs
        image = image.to(device)
        mask = mask.to(device)
        image, image1 = nodel(image)
        output = torch.sigmoid(image).data.cpu().numpy()
        mask = mask.cpu().numpy()
        masks.append(mask)
        predicts.append(output)
        namepath = mask_path[mynum]
        PatchPosition, NameNow = GetPatchPosition(namepath)
        LastName = NameNow
        if len(masks) == 27:
            OneWT = np.zeros([180, 180, 180], dtype=np.uint8)
            OneTC = np.zeros([180, 180, 180], dtype=np.uint8)
            OneET = np.zeros([180, 180, 180], dtype=np.uint8)
            # 创建三个全黑的三维矩阵，分别用于真实的WT、TC、ET分块的拼接
            OneWTMask = np.zeros([180, 180, 180], dtype=np.uint8)
            OneTCMask = np.zeros([180, 180, 180], dtype=np.uint8)
            OneETMask = np.zeros([180, 180, 180], dtype=np.uint8)
            for i in range(27):
                #output=predicts[i]
                mask = masks[i]
                image = predicts[i]
                mask = mask.squeeze(0)
                image = image.squeeze(0)
                #print(mask.shape)
                for idz in range(10, mask.shape[1] - 10):
                    for idx in range(10, mask.shape[2] - 10):
                        for idy in range(10, mask.shape[3] - 10):
                            OneWTMask[box[i][0] * 60 + idz - 10,
                                      box[i][1] * 60 + idx - 10,
                                      box[i][2] * 60 + idy - 10] = mask[0, idz,
                                                                        idx,
                                                                        idy]
                            OneTCMask[box[i][0] * 60 + idz - 10,
                                      box[i][1] * 60 + idx - 10,
                                      box[i][2] * 60 + idy - 10] = mask[1, idz,
                                                                        idx,
                                                                        idy]
                            OneETMask[box[i][0] * 60 + idz - 10,
                                      box[i][1] * 60 + idx - 10,
                                      box[i][2] * 60 + idy - 10] = mask[2, idz,
                                                                        idx,
                                                                        idy]
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
            masks.clear()

            mask = np.zeros([3, 155, 240, 240], dtype=np.uint8)
            prediction = np.zeros([3, 155, 240, 240], dtype=np.uint8)
            mask[0, :, 30:210, 30:210] = OneWTMask[13:168, :, :]
            mask[1, :, 30:210, 30:210] = OneTCMask[13:168, :, :]
            mask[2, :, 30:210, 30:210] = OneETMask[13:168, :, :]
            prediction[0, :, 30:210, 30:210] = OneWT[13:168, :, :]
            prediction[1, :, 30:210, 30:210] = OneTC[13:168, :, :]
            prediction[2, :, 30:210, 30:210] = OneET[13:168, :, :]

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

            dice_scores = dice_coef_metric_per_classes(prediction, mask)
            iou_scores = jaccard_coef_metric_per_classes(prediction, mask)
            sens_scores = sensitivity_metric_per_classes(prediction, mask)
            spec_scores = specificity_metric_per_classes(prediction, mask)
            haus_scores = hausdorff_95_per_classes(prediction, mask)

            for key in dice_scores.keys():
                dice_scores_per_classes[key].extend(dice_scores[key])

            for key in iou_scores.keys():
                iou_scores_per_classes[key].extend(iou_scores[key])

            for key in sens_scores.keys():
                sens_scores_per_classes[key].extend(sens_scores[key])

            for key in spec_scores.keys():
                spec_scores_per_classes[key].extend(spec_scores[key])

            for key in haus_scores.keys():
                haus_scores_per_classes[key].extend(haus_scores[key])

dice_df = pd.DataFrame(dice_scores_per_classes)
dice_df.columns = ['WT dice', 'TC dice', 'ET dice']

iou_df = pd.DataFrame(iou_scores_per_classes)
iou_df.columns = ['WT jaccard', 'TC jaccard', 'ET jaccard']

sens_df = pd.DataFrame(sens_scores_per_classes)
sens_df.columns = ['WT sens', 'TC sens', 'ET sens']

spec_df = pd.DataFrame(spec_scores_per_classes)
spec_df.columns = ['WT spec', 'TC spec', 'ET spec']

haus_df = pd.DataFrame(haus_scores_per_classes)
haus_df.columns = ['WT haus', 'TC haus', 'ET haus']

val_metics_df = pd.concat([dice_df, iou_df, sens_df, spec_df, haus_df],
                          axis=1,
                          sort=True)
val_metics_df = val_metics_df.loc[:, [
    'WT dice', 'WT jaccard', 'WT sens', 'WT spec', 'WT haus', 'TC dice',
    'TC jaccard', 'TC sens', 'TC spec', 'TC haus', 'ET dice', 'ET jaccard',
    'ET sens', 'ET spec', 'ET haus'
]]
print(val_metics_df.mean())
val_metics_df.to_csv("val_metics_df.csv", index=False)
