import torch.nn as nn


class Pix2mesh_Cond(nn.Module):
    r''' Conditioning Network proposed in the authors' Pixel2Mesh implementation.

    The network consists of several 2D convolution layers, and several of the
    intermediate feature maps are returned to features for the image
    projection layer of the encoder network.
    '''
    def __init__(self, c_dim=512, return_feature_maps=True):
        r''' Initialisation.

        Args:
            c_dim (int): channels of the final output
            return_feature_maps (bool): whether intermediate feature maps
                    should be returned
        '''
        super().__init__()
        actvn = nn.ReLU()
        self.return_feature_maps = return_feature_maps
        num_fm = int(c_dim/32)
        if num_fm != 16:
            raise ValueError('Pixel2Mesh requires a fixed c_dim of 512!')

        self.block_1 = nn.Sequential(
            nn.Conv2d(3, num_fm, 3, stride=1, padding=1), actvn,
            nn.Conv2d(num_fm, num_fm, 3, stride=1, padding=1), actvn,
            nn.Conv2d(num_fm, num_fm*2, 3, stride=2, padding=1), actvn,
            nn.Conv2d(num_fm*2, num_fm*2, 3, stride=1, padding=1), actvn,
            nn.Conv2d(num_fm*2, num_fm*2, 3, stride=1, padding=1), actvn,
            nn.Conv2d(num_fm*2, num_fm*4, 3, stride=2, padding=1), actvn,
            nn.Conv2d(num_fm*4, num_fm*4, 3, stride=1, padding=1), actvn,
            nn.Conv2d(num_fm*4, num_fm*4, 3, stride=1, padding=1), actvn)

        self.block_2 = nn.Sequential(
            nn.Conv2d(num_fm*4, num_fm*8, 3, stride=2, padding=1), actvn,
            nn.Conv2d(num_fm*8, num_fm*8, 3, stride=1, padding=1), actvn,
            nn.Conv2d(num_fm*8, num_fm*8, 3, stride=1, padding=1), actvn)

        self.block_3 = nn.Sequential(
            nn.Conv2d(num_fm*8, num_fm*16, 5, stride=2, padding=2), actvn,
            nn.Conv2d(num_fm*16, num_fm*16, 3, stride=1, padding=1), actvn,
            nn.Conv2d(num_fm*16, num_fm*16, 3, stride=1, padding=1), actvn)

        self.block_4 = nn.Sequential(
            nn.Conv2d(num_fm*16, num_fm*32, 5, stride=2, padding=2), actvn,
            nn.Conv2d(num_fm*32, num_fm*32, 3, stride=1, padding=1), actvn,
            nn.Conv2d(num_fm*32, num_fm*32, 3, stride=1, padding=1), actvn,
            nn.Conv2d(num_fm*32, num_fm*32, 3, stride=1, padding=1), actvn,
        )

    def forward(self, x):
        # x has size 224 x 224
        x_0 = self.block_1(x)  # 64 x 56 x 56
        x_1 = self.block_2(x_0)  # 128 x 28 x 28
        x_2 = self.block_3(x_1)  # 256 x 14 x 14
        x_3 = self.block_4(x_2)  # 512 x 7 x 7

        if self.return_feature_maps:
            return x_0, x_1, x_2, x_3
        return x_3
