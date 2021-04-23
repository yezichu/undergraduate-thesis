from model.Scale import Scale, Scale3d
import torch.nn.functional as F
import torch.nn as nn


class conv_block(nn.Sequential):
    def __init__(self,
                 nb_inp_fea,
                 growth_rate,
                 dropout_rate=0,
                 weight_decay=1e-4):
        super(conv_block, self).__init__()
        eps = 1.1e-5
        self.drop = dropout_rate
        self.add_module('norm1', nn.BatchNorm2d(nb_inp_fea,
                                                eps=eps,
                                                momentum=1))
        self.add_module('scale1', Scale(nb_inp_fea))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv2d1',
            nn.Conv2d(nb_inp_fea, 4 * growth_rate, (1, 1), bias=False))
        self.add_module('norm2',
                        nn.BatchNorm2d(4 * growth_rate, eps=eps, momentum=1))
        self.add_module('scale2', Scale(4 * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv2d2',
            nn.Conv2d(4 * growth_rate,
                      growth_rate, (3, 3),
                      padding=(1, 1),
                      bias=False))

    def forward(self, x):
        out = self.norm1(x)
        out = self.scale1(out)
        out = self.relu1(out)
        out = self.conv2d1(out)

        if (self.drop > 0):
            out = F.dropout(out, p=self.drop)

        out = self.norm2(out)
        out = self.scale2(out)
        out = self.relu2(out)
        out = self.conv2d2(out)

        if (self.drop > 0):
            out = F.dropout(out, p=self.drop)

        return out


class conv_block3d(nn.Sequential):
    def __init__(self,
                 nb_inp_fea,
                 growth_rate,
                 dropout_rate=0,
                 weight_decay=1e-4):
        super(conv_block3d, self).__init__()
        eps = 1.1e-5
        self.drop = dropout_rate
        self.add_module('norm1', nn.BatchNorm3d(nb_inp_fea,
                                                eps=eps,
                                                momentum=1))
        self.add_module('scale1', Scale3d(nb_inp_fea))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv3d1',
            nn.Conv3d(nb_inp_fea, 4 * growth_rate, (1, 1, 1), bias=False))
        self.add_module('norm2',
                        nn.BatchNorm3d(4 * growth_rate, eps=eps, momentum=1))
        self.add_module('scale2', Scale3d(4 * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module(
            'conv3d2',
            nn.Conv3d(4 * growth_rate,
                      growth_rate, (3, 3, 3),
                      padding=(1, 1, 1),
                      bias=False))

    def forward(self, x):
        out = self.norm1(x)
        out = self.scale1(out)
        out = self.relu1(out)
        out = self.conv3d1(out)

        if (self.drop > 0):
            out = F.dropout(out, p=self.drop)

        out = self.norm2(out)
        out = self.scale2(out)
        out = self.relu2(out)
        out = self.conv3d2(out)

        if (self.drop > 0):
            out = F.dropout(out, p=self.drop)

        return out
