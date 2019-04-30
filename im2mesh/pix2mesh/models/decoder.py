import torch
import torch.nn as nn
from im2mesh.pix2mesh.layers import (
    GraphConvolution, GraphProjection, GraphUnpooling)


class Decoder(nn.Module):
    r""" Decoder class for Pixel2Mesh Model.

    Args:
        ellipsoid (list): list of helper matrices for the graph convolution
                and pooling layer
        device (PyTorch device): PyTorch device
        hidden_dim (int): The hidden dimension of the graph convolution layers
        feat_dim (int): The dimension of the feature vector obtained from the
                graph projection layer
        coor_dim (int): Output point dimension (usually 3)
        adjust_ellipsoid (bool): whether the ellipsoid should be adjusted by
                inverting the Pixel2Mesh authors' transformation

    """

    def __init__(self, ellipsoid, device=None, hidden_dim=192,
                 feat_dim=1280, coor_dim=3, adjust_ellipsoid=False):
        super(Decoder, self).__init__()
        # Save necessary helper matrices in respective variables
        self.initial_coordinates = torch.tensor(ellipsoid[0]).to(device)
        if adjust_ellipsoid:
            ''' This is the inverse of the operation the Pixel2mesh authors'
            performed to original CAT model; it ensures that the ellipsoid
            has the same size and scale in the not-transformed coordinate
            system we are using. '''
            print("Adjusting ellipsoid.")
            self.initial_coordinates = self.initial_coordinates / 0.57
            self.initial_coordinates[:, 1] = -self.initial_coordinates[:, 1]
            self.initial_coordinates[:, 2] = -self.initial_coordinates[:, 2]

        self.pool_idx_1 = torch.tensor(ellipsoid[4][0]).to(
            device)  # IDs for the first unpooling operation
        self.pool_idx_2 = torch.tensor(ellipsoid[4][1]).to(
            device)  # IDs for the second unpooling operation
        # sparse support matrices for graph convolution; the indices need to
        # be transposed to match pytorch standards
        ell_1 = ellipsoid[1][1]
        e1, e2, e3 = torch.tensor(ell_1[0]).transpose_(
            0, 1), torch.tensor(ell_1[1]), torch.tensor(ell_1[2])
        self.support_1 = torch.sparse.FloatTensor(
            e1.long(), e2, torch.Size(e3)).to(device)
        ell_2 = ellipsoid[2][1]
        e1, e2, e3 = torch.tensor(ell_2[0]).transpose_(
            0, 1), torch.tensor(ell_2[1]), torch.tensor(ell_2[2])
        self.support_2 = torch.sparse.FloatTensor(
            e1.long(), e2, torch.Size(e3)).to(device)
        ell_3 = ellipsoid[3][1]
        e1, e2, e3 = torch.tensor(ell_3[0]).transpose_(
            0, 1), torch.tensor(ell_3[1]), torch.tensor(ell_3[2])
        self.support_3 = torch.sparse.FloatTensor(
            e1.long(), e2, torch.Size(e3)).to(device)

        # The respective layers of the model; Note that some layers with NO
        # weights are reused to save memory
        actvn = nn.ReLU()

        self.gp = GraphProjection()
        self.gc1 = nn.Sequential(GraphConvolution(
            self.support_1, input_dim=feat_dim, output_dim=hidden_dim), actvn)
        self.gc2 = []
        for _ in range(12):
            self.gc2.append(nn.Sequential(GraphConvolution(
                self.support_1, input_dim=hidden_dim, output_dim=hidden_dim),
                actvn))
        self.gc2 = nn.ModuleList(self.gc2)
        self.gc3 = GraphConvolution(
            self.support_1, input_dim=hidden_dim, output_dim=coor_dim)
        self.gup1 = GraphUnpooling(self.pool_idx_1.long())
        self.gc4 = nn.Sequential(GraphConvolution(
            self.support_2, input_dim=feat_dim+hidden_dim,
            output_dim=hidden_dim), actvn)
        self.gc5 = []
        for _ in range(12):
            self.gc5.append(nn.Sequential(GraphConvolution(
                self.support_2, input_dim=hidden_dim, output_dim=hidden_dim),
                actvn))
        self.gc5 = nn.ModuleList(self.gc5)
        self.gc6 = GraphConvolution(
            self.support_2, input_dim=hidden_dim, output_dim=coor_dim)
        self.gup2 = GraphUnpooling(self.pool_idx_2.long())
        self.gc7 = nn.Sequential(GraphConvolution(
            self.support_3, input_dim=feat_dim+hidden_dim,
            output_dim=hidden_dim), actvn)
        self.gc8 = []
        for _ in range(13):
            self.gc8.append(nn.Sequential(GraphConvolution(
                self.support_3, input_dim=hidden_dim, output_dim=hidden_dim),
                actvn))
        self.gc8 = nn.ModuleList(self.gc8)
        self.gc9 = GraphConvolution(
            self.support_3, input_dim=hidden_dim, output_dim=coor_dim)

    def forward(self, x, fm, camera_mat):
        """ Makes a forward pass with the given input through the network.

        Arguments:
            x (tensor): input tensors (e.g. images)
            fm (tensor): feature maps from the conditioned network
            camera_mat (tensor): camera matrices for projection to image plane
        """

        batch_size = x.shape[0]
        # List of initial 3D coordinates (first item) and outputs of the layers
        out = list()

        initial_coordinates_expanded = self.initial_coordinates.expand(
            batch_size, -1, -1)
        out.append(initial_coordinates_expanded)
        
        # #######################
        # First Projection Block
        # Layer 0: 156 x feat_dim
        out.append(self.gp(initial_coordinates_expanded, fm, camera_mat))
        out.append(self.gc1(out[-1]))  # Layer 1: 156 x hidden_dim
        for i in range(0, 12):  # GraphConvs from and to 156 x hidden_dim
            val = self.gc2[i](out[-1])
            if (i % 2) == 1:
                # Add previous output (Restnet style)
                val = torch.add(val, out[-2]) * 0.5
            out.append(val)
        # Layer 14: Out of dim 156x3, will be used as outputs_2[1]
        out.append(self.gc3(out[-1]))

        # #######################
        # Second Projection Block
        # Layer 15: 156 x (hidden_dim + feat_dim)
        v = self.gp(out[-1], fm, camera_mat)
        v = torch.cat([v, out[-2]], dim=2)
        out.append(v)
        # Layer 16: 618x (hidden_dim + feat_dim)
        out.append(self.gup1(out[-1]))
        out.append(self.gc4(out[-1]))  # Layer 17: 618 x hidden_dim
        for i in range(0, 12):  # GraphConvs from and to 618 x hidden_dim
            val = self.gc5[i](out[-1])
            if (i % 2) == 1:
                # Add previous output (Restnet style)
                val = torch.add(val, out[-2]) * 0.5
            out.append(val)
        # Layer 30: 618 x 3, will be used as outputs_2[2]
        out.append(self.gc6(out[-1]))

        # #######################
        # Third Projection Block
        # Layer 31: 618 x hidden_dim + feat_dim
        v = self.gp(out[-1], fm, camera_mat)  # 618 x feat_dim
        v = torch.cat([v, out[-2]], dim=2)
        out.append(v)
        # Layer 32: 2466 x hidden_dim + feat_dim
        out.append(self.gup2(out[-1]))
        out.append(self.gc7(out[-1]))  # Layer 33: 2466 x hidden_dim
        for i in range(0, 13):  # GraphConvs from and to 2466 x hidden_dim
            val = self.gc8[i](out[-1])
            if i % 2 == 1:
                # Add previous output (Restnet style)
                val = torch.add(val, out[-2]) * 0.5
            out.append(val)
        out.append(self.gc9(out[-1]))  # Layer 47: 2466 x 3
        # 156 x hidden_dim, 618 x hidden_dim, 2466 x hidden_dim
        outputs = (out[15], out[31], out[-1])
        outputs_2 = (initial_coordinates_expanded,
                     self.gup1(out[15]), self.gup2(out[31]))

        return outputs, outputs_2
