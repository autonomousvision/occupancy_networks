import torch.nn as nn
import torch
from im2mesh.dmc.ops.occupancy_to_topology import OccupancyToTopology


class UNetDecoder(nn.Module):
    def __init__(self, input_dim=16, T=256, W=32, H=32, D=32, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        self.decoder = SurfaceDecoder(T, W, H, D, skip_connection)
        self.encoder = LocalEncoder(input_dim, skip_connection)

    def forward(self, c):
        if self.skip_connection:
            z, intermediate_feat = self.encoder(c)
            occupancy, offset, topology = self.decoder(z, intermediate_feat)
        else:
            z = self.encoder(c)
            occupancy, offset, topology = self.decoder(z)

        return offset, topology, occupancy


class SurfaceDecoder(nn.Module):
    """Decoder of the U-Net, estimate topology and offset with two headers"""
    def __init__(self,  T=256, W=32, H=32, D=32, skip_connection=True):
        super(SurfaceDecoder, self).__init__()
        self.W = W
        self.H = H
        self.D = D
        self.T = T

        self.actvn = nn.ReLU()
        self.Occ2Top = OccupancyToTopology()

        # decoder
        self.deconv4 = nn.Conv3d(128, 64, 3, padding=1)
        self.deconv3_1 = nn.ConvTranspose3d(128, 128, 3, padding=1)
        self.deconv3_2 = nn.ConvTranspose3d(128, 32, 3, padding=1)
        self.deconv2_off_1 = nn.ConvTranspose3d(64, 64, 3, padding=1)
        self.deconv2_off_2 = nn.ConvTranspose3d(64, 16, 3, padding=1)
        self.deconv2_occ_1 = nn.ConvTranspose3d(64, 64, 3, padding=1)
        self.deconv2_occ_2 = nn.ConvTranspose3d(64, 16, 3, padding=1)
        self.deconv1_off_1 = nn.ConvTranspose3d(32, 32, 3, padding=1)
        self.deconv1_off_2 = nn.ConvTranspose3d(32, 3, 3, padding=3)
        self.deconv1_occ_1 = nn.ConvTranspose3d(32, 32, 3, padding=1)
        self.deconv1_occ_2 = nn.ConvTranspose3d(32, 1, 3, padding=3)
        
        # batchnorm
        self.deconv4_bn = nn.BatchNorm3d(64)
        self.deconv3_1_bn = nn.BatchNorm3d(128)
        self.deconv3_2_bn = nn.BatchNorm3d(32)
        self.deconv2_off_1_bn = nn.BatchNorm3d(64)
        self.deconv2_off_2_bn = nn.BatchNorm3d(16)
        self.deconv2_occ_1_bn = nn.BatchNorm3d(64)
        self.deconv2_occ_2_bn = nn.BatchNorm3d(16)
        self.deconv1_off_1_bn = nn.BatchNorm3d(32)
        self.deconv1_occ_1_bn = nn.BatchNorm3d(32)

        self.sigmoid = nn.Sigmoid()

        self.maxunpool = nn.MaxUnpool3d(2)

        self.skip_connection = skip_connection

    def decoder(self, x, intermediate_feat=None):

        if self.skip_connection:
            feat1, size1, indices1, feat2, size2, indices2, feat3, size3, indices3 = intermediate_feat

        #
        x = self.actvn(self.deconv4_bn(self.deconv4(x)))

        #
        x = self.maxunpool(x, indices3, output_size=size3)
        if self.skip_connection:
            x = torch.cat((feat3, x), 1)
        x = self.actvn(self.deconv3_1_bn(self.deconv3_1(x)))
        x = self.actvn(self.deconv3_2_bn(self.deconv3_2(x)))

        #
        x = self.maxunpool(x, indices2, output_size=size2)
        if self.skip_connection:
            x = torch.cat((feat2, x), 1)
        x_occupancy = self.actvn(self.deconv2_occ_1_bn(self.deconv2_occ_1(x)))
        x_occupancy = self.actvn(
            self.deconv2_occ_2_bn(self.deconv2_occ_2(x_occupancy)))
        x_offset = self.actvn(self.deconv2_off_1_bn(self.deconv2_off_1(x)))
        x_offset = self.actvn(
            self.deconv2_off_2_bn(self.deconv2_off_2(x_offset)))

        #
        x_occupancy = self.maxunpool(x_occupancy, indices1, output_size=size1)
        if self.skip_connection:
            x_occupancy = torch.cat((feat1, x_occupancy), 1)
        x_offset = self.maxunpool(x_offset, indices1, output_size=size1)
        if self.skip_connection:
            x_offset = torch.cat((feat1, x_offset), 1)
        x_occupancy = self.actvn(
            self.deconv1_occ_1_bn(self.deconv1_occ_1(x_occupancy)))
        x_occupancy = self.sigmoid(self.deconv1_occ_2(x_occupancy))
        x_offset = self.actvn(
            self.deconv1_off_1_bn(self.deconv1_off_1(x_offset)))
        x_offset = self.sigmoid(self.deconv1_off_2(x_offset)) - 0.5

        batchsize = x_occupancy.size()[0]
        topology = torch.zeros(batchsize, self.W*self.H*self.D, self.T).to(x.device)

        for k in range(batchsize):
            topology[k, :, :] = self.Occ2Top(x_occupancy[k, 0, :, :])

        return x_occupancy, x_offset, topology

    def forward(self, x, intermediate_feat=None):
        return self.decoder(x, intermediate_feat)


class LocalEncoder(nn.Module):
    """Encoder of the U-Net"""
    def __init__(self, input_dim=16, skip_connection=True):
        super(LocalEncoder, self).__init__()

        self.actvn = nn.ReLU()

        # u-net
        self.conv1_1 = nn.Conv3d(input_dim, 16, 3, padding=3)
        self.conv1_2 = nn.Conv3d(16, 16, 3, padding=1)
        self.conv2_1 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv2_2 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv3_1 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3_2 = nn.Conv3d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv3d(64, 128, 3, padding=1)

        # batchnorm
        self.conv1_1_bn = nn.BatchNorm3d(16)
        self.conv1_2_bn = nn.BatchNorm3d(16)
        self.conv2_1_bn = nn.BatchNorm3d(32)
        self.conv2_2_bn = nn.BatchNorm3d(32)
        self.conv3_1_bn = nn.BatchNorm3d(64)
        self.conv3_2_bn = nn.BatchNorm3d(64)
        self.conv4_bn = nn.BatchNorm3d(128)

        self.maxpool = nn.MaxPool3d(2, return_indices=True)

        self.skip_connection = skip_connection

    def encoder(self, x):
        x = self.actvn(self.conv1_1_bn(self.conv1_1(x)))
        x = self.actvn(self.conv1_2_bn(self.conv1_2(x)))
        feat1 = x
        size1 = x.size()
        x, indices1 = self.maxpool(x)

        #
        x = self.actvn(self.conv2_1_bn(self.conv2_1(x)))
        x = self.actvn(self.conv2_2_bn(self.conv2_2(x)))
        feat2 = x
        size2 = x.size()
        x, indices2 = self.maxpool(x)

        #
        x = self.actvn(self.conv3_1_bn(self.conv3_1(x)))
        x = self.actvn(self.conv3_2_bn(self.conv3_2(x)))
        feat3 = x
        size3 = x.size()
        x, indices3 = self.maxpool(x)

        #
        x = self.actvn(self.conv4_bn(self.conv4(x)))
        return x, feat1, size1, indices1, feat2, size2, indices2, feat3, size3, indices3

    def forward(self, x):
        x, feat1, size1, indices1, feat2, size2, indices2, feat3, size3, indices3 = self.encoder(x)
        if self.skip_connection:
            return x, (feat1, size1, indices1, feat2, size2, indices2, feat3, size3, indices3)
        else:
            return x