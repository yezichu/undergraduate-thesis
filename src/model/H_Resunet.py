from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
# from denseUnet import denseUnet
from model.Resatten import ResUnet
# from denseUnet3d import denseUnet3d
from model.DMFNet_16x import DMFNet

device = 'cuda'


class H_ResUnet(nn.Module):
    def __init__(self, num_slide, drop_rate=0):
        super(H_ResUnet, self).__init__()
        self.num_slide = num_slide
        self.drop = drop_rate
        self.dense2d = ResUnet()
        # self.dense3d = denseUnet3d(7)
        self.dense3d = DMFNet(c=7, groups=16, norm='gn')
        self.conv2d5 = nn.Conv2d(64, 3, (1, 1), padding=0)
        self.conv3d5 = nn.Conv3d(64, 3, (1, 1, 1), padding=0)
        self.finalConv3d1 = nn.Conv3d(64, 64, (3, 3, 3), padding=(1, 1, 1))
        self.finalBn = nn.GroupNorm(num_groups=4, num_channels=64)
        self.finalAc = nn.ReLU(inplace=True)
        self.finalConv3d2 = nn.Conv3d(64, 3, (1, 1, 1))


    def forward(self, x):
        x1 = x.clone().squeeze(0)
        x1 = x1.permute(1, 0, 2, 3)
        """
        input2d = x[:, :, 0:2, :, :]

        single = x[:, :, 0:1, :, :]
        input2d = torch.cat((input2d, single), 2)
        for i in range(self.num_slide - 2):
            input2dtmp = x[:, :, i:i + 3, :, :]
            input2d = torch.cat((input2d, input2dtmp), 0)
            if i == self.num_slide - 3:
                f1 = x[:, :, self.num_slide - 2:self.num_slide, :, :]
                f2 = x[:, :, self.num_slide - 1:self.num_slide, :, :]
                ff = torch.cat((f1, f2), 2)
                input2d = torch.cat((input2d, ff), 0)
        input2d = input2d[:, 0:1, :, :, :]
        input2d = input2d.squeeze(1)
        # input2d = input2d[:, :, :, :, 0]
        # input2d = input2d.permute(0, 3, 1, 2)
        """
        feature2d = self.dense2d(x1)
        final2d = self.conv2d5(feature2d)

        input3d = final2d.clone().permute(1, 0, 2, 3)
        feature2d = feature2d.clone().permute(1, 0, 2, 3)
        input3d1 = input3d.unsqueeze(0)
        feature2d = feature2d.unsqueeze(0)
        x_tmp = x.clone()
        input3d = torch.cat((input3d1, x_tmp), 1)
        feature3d = self.dense3d(input3d)
        final = torch.add(feature2d, feature3d) / 2
        finalout = self.finalConv3d1(final)
        if (self.drop > 0):
            finalout = F.dropout(finalout, p=self.drop)

        finalout = self.finalBn(finalout)
        finalout = self.finalAc(finalout)
        finalout = self.finalConv3d2(finalout)

        return finalout, input3d1
