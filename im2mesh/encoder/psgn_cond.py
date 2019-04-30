import torch.nn as nn


class PCGN_Cond(nn.Module):
    r''' Point Set Generation Network encoding network.

    The PSGN conditioning network from the original publication consists of
    several 2D convolution layers. The intermediate outputs from some layers
    are used as additional input to the encoder network, similar to U-Net.

    Args:
        c_dim (int): output dimension of the latent embedding
    '''
    def __init__(self, c_dim=512):  
        super().__init__()
        actvn = nn.ReLU()
        num_fm = int(c_dim/32)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, num_fm, 3, 1, 1), actvn,
            nn.Conv2d(num_fm, num_fm, 3, 1, 1), actvn)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(num_fm, num_fm*2, 3, 2, 1), actvn,
            nn.Conv2d(num_fm*2, num_fm*2, 3, 1, 1), actvn,
            nn.Conv2d(num_fm*2, num_fm*2, 3, 1, 1), actvn)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(num_fm*2, num_fm*4, 3, 2, 1), actvn,
            nn.Conv2d(num_fm*4, num_fm*4, 3, 1, 1), actvn,
            nn.Conv2d(num_fm*4, num_fm*4, 3, 1, 1), actvn)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(num_fm*4, num_fm*8, 3, 2, 1), actvn,
            nn.Conv2d(num_fm*8, num_fm*8, 3, 1, 1), actvn,
            nn.Conv2d(num_fm*8, num_fm*8, 3, 1, 1), actvn)
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(num_fm*8, num_fm*16, 3, 2, 1), actvn,
            nn.Conv2d(num_fm*16, num_fm*16, 3, 1, 1), actvn,
            nn.Conv2d(num_fm*16, num_fm*16, 3, 1, 1), actvn)
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(num_fm*16, num_fm*32, 3, 2, 1), actvn,
            nn.Conv2d(num_fm*32, num_fm*32, 3, 1, 1), actvn,
            nn.Conv2d(num_fm*32, num_fm*32, 3, 1, 1), actvn,
            nn.Conv2d(num_fm*32, num_fm*32, 3, 1, 1), actvn)
        self.conv_block7 = nn.Sequential(
            nn.Conv2d(num_fm*32, num_fm*32, 5, 2, 2), actvn)

        self.trans_conv1 = nn.Conv2d(num_fm*8, num_fm*4, 3, 1, 1)
        self.trans_conv2 = nn.Conv2d(num_fm*16, num_fm*8, 3, 1, 1)
        self.trans_conv3 = nn.Conv2d(num_fm*32, num_fm*16, 3, 1, 1)

    def forward(self, x, return_feature_maps=True):
        r''' Performs a forward pass through the network.

        Args:
            x (tensor): input data
            return_feature_maps (bool): whether intermediate feature maps
                    should be returned
        '''
        feature_maps = []

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        feature_maps.append(self.trans_conv1(x))

        x = self.conv_block5(x)
        feature_maps.append(self.trans_conv2(x))

        x = self.conv_block6(x)
        feature_maps.append(self.trans_conv3(x))

        x = self.conv_block7(x)

        if return_feature_maps:
            return x, feature_maps
        return x
