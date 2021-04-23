import torch
import torch.nn as nn
from model.conv_block import conv_block, conv_block3d


class dense_block(nn.Module):
    def __init__(self,
                 nb_layers,
                 nb_filter,
                 growth_rate,
                 dropout_rate=0,
                 weight_decay=1e-4,
                 grow_nb_filters=True):
        super(dense_block, self).__init__()
        for i in range(nb_layers):
            layer = conv_block(nb_filter + i * growth_rate, growth_rate,
                               dropout_rate)
            self.add_module('denseLayer%d' % (i + 1), layer)

    def forward(self, x):
        features = [x]
        for name, layer in self.named_children():
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)


class dense_block3d(nn.Module):
    def __init__(self,
                 nb_layers,
                 nb_filter,
                 growth_rate,
                 dropout_rate=0,
                 weight_decay=1e-4,
                 grow_nb_filters=True):
        super(dense_block3d, self).__init__()
        for i in range(nb_layers):
            layer = conv_block3d(nb_filter + i * growth_rate, growth_rate,
                                 dropout_rate)
            self.add_module('denseLayer3d%d' % (i + 1), layer)

    def forward(self, x):
        features = [x]
        for name, layer in self.named_children():
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        return torch.cat(features, 1)
