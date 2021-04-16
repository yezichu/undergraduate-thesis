import torch
import torch.nn as nn


class Scale(nn.Module):
    def __init__(self, num_feature):
        super(Scale, self).__init__()
        self.num_feature = num_feature
        self.gamma = nn.Parameter(torch.ones(num_feature), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(num_feature), requires_grad=True)

    def forward(self, x):
        y = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for i in range(self.num_feature):
            y[:,
              i, :, :] = x[:, i, :, :].clone() * self.gamma[i] + self.beta[i]
        return y


class Scale3d(nn.Module):
    def __init__(self, num_feature):
        super(Scale3d, self).__init__()
        self.num_feature = num_feature
        self.gamma = nn.Parameter(torch.ones(num_feature), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(num_feature), requires_grad=True)

    def forward(self, x):
        y = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for i in range(self.num_feature):
            y[:, i, :, :, :] = x[:, i, :, :, :].clone(
            ) * self.gamma[i] + self.beta[i]
        return y
