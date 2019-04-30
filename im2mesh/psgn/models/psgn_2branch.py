import torch.nn as nn
import torch


class PCGN_2Branch(nn.Module):
    r''' The 2-Branch decoder of the Point Set Generation Network.

    The latent embedding of the image is passed through a fully-connected
    branch as well as a convolution-based branch which receives additional
    input from the conditioning network.
    '''
    def __init__(self, dim=3, c_dim=512, n_points=1024):
        r''' Initialisation.

        Args:
            dim (int): dimension of the output points (e.g. 3)
            c_dim (int): dimension of the output of the conditioning network
            n_points (int): number of points to predict

        '''
        super().__init__()
        # Attributes
        actvn = nn.ReLU()
        self.actvn = actvn
        self.dim = dim
        num_fm = int(c_dim/32)
        conv_c_in = 32 * num_fm
        fc_dim_in = 3*4*conv_c_in  # input image is downsampled to 3x4
        fc_pts = n_points - 768  # conv branch has a fixed output of 768 points

        # Submodules
        self.fc_branch = nn.Sequential(nn.Linear(fc_dim_in, fc_pts*dim), actvn)
        self.deconv_1 = nn.ConvTranspose2d(c_dim, num_fm*16, 5, 2, 2, 1)
        self.deconv_2 = nn.ConvTranspose2d(num_fm*16, num_fm*8, 5, 2, 2, 1)
        self.deconv_3 = nn.ConvTranspose2d(num_fm*8, num_fm*4, 5, 2, 2, 1)
        # TODO: unused, remove? (keep it for now to load old checkpoints)
        self.deconv_4 = nn.ConvTranspose2d(num_fm*4, 3, 5, 2, 2, 1)

        self.conv_1 = nn.Sequential(
            nn.Conv2d(num_fm*16, num_fm*16, 3, 1, 1), actvn)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(num_fm*8, num_fm*8, 3, 1, 1), actvn)
        self.conv_3 = nn.Sequential(
            nn.Conv2d(num_fm*4, num_fm*4, 3, 1, 1), actvn)
        self.conv_4 = nn.Conv2d(num_fm*4, dim, 3, 1, 1)

    def forward(self, c):
        x, feature_maps = c
        batch_size = x.shape[0]

        fc_branch = self.fc_branch(x.view(batch_size, -1))
        fc_branch = fc_branch.view(batch_size, -1, 3)

        conv_branch = self.deconv_1(x)
        conv_branch = self.actvn(torch.add(conv_branch, feature_maps[-1]))

        conv_branch = self.conv_1(conv_branch)
        conv_branch = self.deconv_2(conv_branch)
        conv_branch = self.actvn(torch.add(conv_branch, feature_maps[-2]))

        conv_branch = self.conv_2(conv_branch)
        conv_branch = self.deconv_3(conv_branch)
        conv_branch = self.actvn(torch.add(conv_branch, feature_maps[-3]))

        conv_branch = self.conv_3(conv_branch)
        conv_branch = self.conv_4(conv_branch)
        conv_branch = conv_branch.view(batch_size, -1, self.dim)

        output = torch.cat([fc_branch, conv_branch], dim=1)
        return output
