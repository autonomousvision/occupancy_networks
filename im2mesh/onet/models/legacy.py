import torch
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import ResnetBlockFC, AffineLayer


class VoxelDecoder(nn.Module):
    def __init__(self, dim=3, z_dim=128, c_dim=128, hidden_size=128):
        super().__init__()
        self.c_dim = c_dim
        self.z_dim = z_dim
        # Submodules
        self.actvn = F.relu
        # 3D decoder
        self.fc_in = nn.Linear(c_dim + z_dim, 256*4*4*4)
        self.convtrp_0 = nn.ConvTranspose3d(256, 128, 3, stride=2,
                                            padding=1, output_padding=1)
        self.convtrp_1 = nn.ConvTranspose3d(128, 64, 3, stride=2,
                                            padding=1, output_padding=1)
        self.convtrp_2 = nn.ConvTranspose3d(64, 32, 3, stride=2,
                                            padding=1, output_padding=1)
        # Fully connected decoder
        self.z_dim = z_dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        self.fc_f = nn.Linear(32, hidden_size)
        self.fc_c = nn.Linear(c_dim, hidden_size)
        self.fc_p = nn.Linear(dim, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size, hidden_size)
        self.block1 = ResnetBlockFC(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, p, z, c, **kwargs):
        batch_size = c.size(0)

        if self.z_dim != 0:
            net = torch.cat([z, c], dim=1)
        else:
            net = c

        net = self.fc_in(net)
        net = net.view(batch_size, 256, 4, 4, 4)
        net = self.convtrp_0(self.actvn(net))
        net = self.convtrp_1(self.actvn(net))
        net = self.convtrp_2(self.actvn(net))

        net = F.grid_sample(
            net, 2*p.unsqueeze(1).unsqueeze(1), padding_mode='border')
        net = net.squeeze(2).squeeze(2).transpose(1, 2)
        net = self.fc_f(self.actvn(net))

        net_p = self.fc_p(p)
        net = net + net_p

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(1)
            net = net + net_z

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(1)
            net = net + net_c

        net = self.block0(net)
        net = self.block1(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class FeatureDecoder(nn.Module):
    def __init__(self, dim=3, z_dim=128, c_dim=128, hidden_size=256):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim

        self.actvn = nn.ReLU()

        self.affine = AffineLayer(c_dim, dim)
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        self.fc_p1 = nn.Linear(dim, hidden_size)
        self.fc_p2 = nn.Linear(dim, hidden_size)

        self.fc_c1 = nn.Linear(c_dim, hidden_size)
        self.fc_c2 = nn.Linear(c_dim, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size, hidden_size)
        self.block1 = ResnetBlockFC(hidden_size, hidden_size)
        self.block2 = ResnetBlockFC(hidden_size, hidden_size)
        self.block3 = ResnetBlockFC(hidden_size, hidden_size)

        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, p, z, c, **kwargs):
        batch_size, T, D = p.size()

        c1 = c.view(batch_size, self.c_dim, -1).max(dim=2)[0]
        Ap = self.affine(c1, p)
        Ap2 = Ap[:, :, :2] / (Ap[:, :, 2:].abs() + 1e-5)

        c2 = F.grid_sample(c, 2*Ap2.unsqueeze(1), padding_mode='border')
        c2 = c2.squeeze(2).transpose(1, 2)

        net = self.fc_p1(p) + self.fc_p2(Ap)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(1)
            net = net + net_z

        net_c = self.fc_c2(c2) + self.fc_c1(c1).unsqueeze(1)
        net = net + net_c

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out