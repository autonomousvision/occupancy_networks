import torch.nn as nn
# import torch.nn.functional as F
from im2mesh.common import normalize_imagenet


class SimpleConv(nn.Module):
    '''  3D Recurrent Reconstruction Neural Network (3D-R2-N2) encoder network.

    Args:
        c_dim: output dimension
    '''

    def __init__(self, c_dim=1024):
        super().__init__()
        actvn = nn.LeakyReLU()
        pooling = nn.MaxPool2d(2, padding=1)
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 96, 7, padding=3),
            pooling, actvn,
            nn.Conv2d(96, 128, 3, padding=1),
            pooling, actvn,
            nn.Conv2d(128, 256, 3, padding=1),
            pooling, actvn,
            nn.Conv2d(256, 256, 3, padding=1),
            pooling, actvn,
            nn.Conv2d(256, 256, 3, padding=1),
            pooling, actvn,
            nn.Conv2d(256, 256, 3, padding=1),
            pooling, actvn,
        )
        self.fc_out = nn.Linear(256*3*3, c_dim)

    def forward(self, x):
        batch_size = x.size(0)

        net = normalize_imagenet(x)
        net = self.convnet(net)
        net = net.view(batch_size, 256*3*3)
        out = self.fc_out(net)

        return out


class Resnet(nn.Module):
    '''  3D Recurrent Reconstruction Neural Network (3D-R2-N2) ResNet-based
        encoder network.

    It is the ResNet variant of the previous encoder.s

    Args:
        c_dim: output dimension
    '''

    def __init__(self, c_dim=1024):
        super().__init__()
        actvn = nn.LeakyReLU()
        pooling = nn.MaxPool2d(2, padding=1)
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 96, 7, padding=3),
            actvn,
            nn.Conv2d(96, 96, 3, padding=1),
            actvn, pooling,
            ResnetBlock(96, 128),
            pooling,
            ResnetBlock(128, 256),
            pooling,
            ResnetBlock(256, 256),
            pooling,
            ResnetBlock(256, 256),
            pooling,
            ResnetBlock(256, 256),
            pooling,
        )
        self.fc_out = nn.Linear(256*3*3, c_dim)

    def forward(self, x):
        batch_size = x.size(0)

        net = normalize_imagenet(x)
        net = self.convnet(net)
        net = net.view(batch_size, 256*3*3)
        out = self.fc_out(net)

        return out


class ResnetBlock(nn.Module):
    ''' ResNet block class.

    Args:
        f_in (int): input dimension
        f_out (int): output dimension
    '''

    def __init__(self, f_in, f_out):
        super().__init__()
        actvn = nn.LeakyReLU()
        self.convnet = nn.Sequential(
            nn.Conv2d(f_in, f_out, 3, padding=1),
            actvn,
            nn.Conv2d(f_out, f_out, 3, padding=1),
            actvn,
        )
        self.shortcut = nn.Conv2d(f_in, f_out, 1)

    def forward(self, x):
        out = self.convnet(x) + self.shortcut(x)
        return out
