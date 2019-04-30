import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelEncoder(nn.Module):
    ''' 3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): output dimension
    '''

    def __init__(self, dim=3, c_dim=128):
        super().__init__()
        self.actvn = F.relu

        self.conv_in = nn.Conv3d(1, 32, 3, padding=1)

        self.conv_0 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        self.conv_1 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv_2 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.conv_3 = nn.Conv3d(256, 512, 3, padding=1, stride=2)
        self.fc = nn.Linear(512 * 2 * 2 * 2, c_dim)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.unsqueeze(1)
        net = self.conv_in(x)
        net = self.conv_0(self.actvn(net))
        net = self.conv_1(self.actvn(net))
        net = self.conv_2(self.actvn(net))
        net = self.conv_3(self.actvn(net))

        hidden = net.view(batch_size, 512 * 2 * 2 * 2)
        c = self.fc(self.actvn(hidden))

        return c


class CoordVoxelEncoder(nn.Module):
    ''' 3D-convolutional encoder network for voxel input.

    It additional concatenates the coordinate data.

    Args:
        dim (int): input dimension
        c_dim (int): output dimension
    '''

    def __init__(self, dim=3, c_dim=128):
        super().__init__()
        self.actvn = F.relu

        self.conv_in = nn.Conv3d(4, 32, 3, padding=1)

        self.conv_0 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        self.conv_1 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv_2 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.conv_3 = nn.Conv3d(256, 512, 3, padding=1, stride=2)
        self.fc = nn.Linear(512 * 2 * 2 * 2, c_dim)

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        coord1 = torch.linspace(-0.5, 0.5, x.size(1)).to(device)
        coord2 = torch.linspace(-0.5, 0.5, x.size(2)).to(device)
        coord3 = torch.linspace(-0.5, 0.5, x.size(3)).to(device)

        coord1 = coord1.view(1, -1, 1, 1).expand_as(x)
        coord2 = coord2.view(1, 1, -1, 1).expand_as(x)
        coord3 = coord3.view(1, 1, 1, -1).expand_as(x)

        coords = torch.stack([coord1, coord2, coord3], dim=1)

        x = x.unsqueeze(1)
        net = torch.cat([x, coords], dim=1)
        net = self.conv_in(net)
        net = self.conv_0(self.actvn(net))
        net = self.conv_1(self.actvn(net))
        net = self.conv_2(self.actvn(net))
        net = self.conv_3(self.actvn(net))

        hidden = net.view(batch_size, 512 * 2 * 2 * 2)
        c = self.fc(self.actvn(hidden))

        return c
