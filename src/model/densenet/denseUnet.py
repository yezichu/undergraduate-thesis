import torch.nn as nn
from torch import cat
from Scale import Scale
from collections import OrderedDict
from dense_block import dense_block
from _Transition import _Transition
from attention import Attention_2dblock


class denseUnet(nn.Module):
    def __init__(self,
                 growth_rate=48,
                 block_config=(6, 12, 36, 24),
                 num_init_features=96,
                 drop_rate=0,):
        super(denseUnet, self).__init__()
        nb_filter = num_init_features
        eps = 1.1e-5

        self.features = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv2d(3,
                           nb_filter,
                           kernel_size=7,
                           stride=2,
                           padding=3,
                           bias=False)),
                ('norm0', nn.BatchNorm2d(nb_filter, eps=eps)),
                ('scale0', Scale(nb_filter)),
                ('relu0', nn.ReLU(inplace=True)),
            ]))

        self.features1 = nn.Sequential(
            OrderedDict([
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))

        self.denseblock1d = dense_block(6, nb_filter, growth_rate, drop_rate)
        nb_filter += 6 * growth_rate
        self.transition1d = _Transition(nb_filter, nb_filter // 2)
        nb_filter = nb_filter // 2

        self.denseblock2d = dense_block(12, nb_filter, growth_rate, drop_rate)
        nb_filter += 12 * growth_rate
        self.transition2d = _Transition(nb_filter, nb_filter // 2)
        nb_filter = nb_filter // 2

        self.denseblock3d = dense_block(36, nb_filter, growth_rate, drop_rate)
        nb_filter += 36 * growth_rate
        self.transition3d = _Transition(nb_filter, nb_filter // 2)
        nb_filter = nb_filter // 2

        self.denseblock4d = dense_block(24, nb_filter, growth_rate, drop_rate)
        nb_filter += 24 * growth_rate

        self.features2 = nn.Sequential(
            OrderedDict([
                ('norm5', nn.BatchNorm2d(nb_filter, eps=eps, momentum=1)),
                ('scale5', Scale(nb_filter)),
                ('relu5', nn.ReLU(inplace=True)),
            ]))

        self.up0 = nn.Upsample(scale_factor=2)

        self.line0 = nn.Conv2d(2112, 2112, (1, 1))
        self.atten0 = Attention_2dblock(F_g=2208, F_l=2112, F_int=768)
        self.decode0 = nn.Sequential(
            OrderedDict([('conv2d0', nn.Conv2d(4320, 768, (3, 3), padding=1)),
                         ('bn0', nn.BatchNorm2d(768, momentum=1)),
                         ('ac0', nn.ReLU(inplace=True))]))

        self.up1 = nn.Upsample(scale_factor=2)
        self.atten1 = Attention_2dblock(F_g=768, F_l=768, F_int=384)
        self.decode1 = nn.Sequential(
            OrderedDict([('conv2d1', nn.Conv2d(1536, 384, (3, 3), padding=1)),
                         ('bn1', nn.BatchNorm2d(384, momentum=1)),
                         ('ac1', nn.ReLU(inplace=True))]))

        self.up2 = nn.Upsample(scale_factor=2)
        self.atten2 = Attention_2dblock(F_g=384, F_l=384, F_int=96)
        self.decode2 = nn.Sequential(
            OrderedDict([('conv2d2', nn.Conv2d(768, 96, (3, 3), padding=1)),
                         ('bn2', nn.BatchNorm2d(96, momentum=1)),
                         ('ac2', nn.ReLU(inplace=True))]))

        self.up3 = nn.Upsample(scale_factor=2)
        self.atten3 = Attention_2dblock(F_g=96, F_l=96, F_int=64)
        self.decode3 = nn.Sequential(
            OrderedDict([('conv2d3', nn.Conv2d(192, 96, (3, 3), padding=1)),
                         ('bn3', nn.BatchNorm2d(96, momentum=1)),
                         ('ac3', nn.ReLU(inplace=True))]))

        self.up4 = nn.Upsample(scale_factor=2)
        self.decode4 = nn.Sequential(
            OrderedDict([('conv2d4', nn.Conv2d(96, 64, (3, 3), padding=1)),
                         ('bn4', nn.BatchNorm2d(64, momentum=1)),
                         ('ac4', nn.ReLU(inplace=True))]))

    def forward(self, x):
        out0 = self.features(x)
        out = self.features1(out0)
        out1 = self.denseblock1d(out)
        out = self.transition1d(out1)
        out2 = self.denseblock2d(out)
        out = self.transition2d(out2)
        out3 = self.denseblock3d(out)
        out = self.transition3d(out3)
        out = self.denseblock4d(out)
        out = self.features2(out)

        out = self.up0(out)
        out3 = self.line0(out3)
        out3 = self.atten0(out, out3)
        out = cat((out, out3), dim=1)
        out = self.decode0(out)
        out = self.up1(out)
        out2 = self.atten1(out, out2)
        out = cat((out, out2), dim=1)
        out = self.decode1(out)
        out = self.up2(out)
        out1 = self.atten2(out, out1)
        out = cat((out, out1), dim=1)
        out = self.decode2(out)
        out = self.up3(out)
        out0 = self.atten3(out, out0)
        out = cat((out, out0), dim=1)
        out = self.decode3(out)
        out = self.up4(out)
        out = self.decode4(out)
        return out
