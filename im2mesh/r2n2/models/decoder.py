import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    ''' Decoder network class for the R2N2 model.

    It consists of 4 transposed 3D-convolutional layers.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
    '''

    def __init__(self, dim=3, c_dim=128):
        super().__init__()
        self.actvn = F.relu
        self.fc_in = nn.Linear(c_dim, 256*4*4*4)
        self.convtrp_0 = nn.ConvTranspose3d(256, 128, 3, stride=2,
                                            padding=1, output_padding=1)
        self.convtrp_1 = nn.ConvTranspose3d(128, 64, 3, stride=2,
                                            padding=1, output_padding=1)
        self.convtrp_2 = nn.ConvTranspose3d(64, 32, 3, stride=2,
                                            padding=1, output_padding=1)
        self.conv_out = nn.Conv3d(32, 1, 1)

    def forward(self, c):
        batch_size = c.size(0)

        net = self.fc_in(c)
        net = net.view(batch_size, 256, 4, 4, 4)
        net = self.convtrp_0(self.actvn(net))
        net = self.convtrp_1(self.actvn(net))
        net = self.convtrp_2(self.actvn(net))

        occ_hat = self.conv_out(self.actvn(net))

        return occ_hat
