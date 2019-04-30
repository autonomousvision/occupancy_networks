import torch.nn as nn
import torch

from im2mesh.dmc.ops.grid_pooling import GridPooling


class PointNetLocal(nn.Module):
    ''' Point Net Local Conditional Network from the Deep Marching Cubes paper.

        It applies two fully connected layers to the input points (dim 3) in a
        1D Convolutional Layer fashion to avoid to specify the number of
        incoming points
    '''
    def __init__(self, c_dim=256, out_dim=16, cell_W=16, cell_H=16, cell_D=16):
        super().__init__()
        self.cell_W = cell_W
        self.cell_H = cell_H
        self.cell_D = cell_D

        # TODO change gridpooling input to be compatible to single values of W H D 
        self.gridshape = torch.cuda.LongTensor([cell_W, cell_H, cell_D])
        actvn = nn.ReLU()
        self.grid_pool = GridPooling(self.gridshape)
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, c_dim, 1), actvn
        )
        #self.conv2 = nn.Sequential(
        #    nn.Conv1d(c_dim, out_dim, 1), actvn
        #)
        self.conv2 = nn.Conv1d(c_dim, out_dim, 1)

    def forward(self, x):
        pts = x
        feats = x.transpose(1, 2)  # b_size x 3 x num_points
        feats = self.conv1(feats)   # b_size x c_dim x num_points
        feats = self.conv2(feats)  # b_size x out_dim x num_points
        feats = feats.transpose(1, 2)  # b_size x num_points x out_dim

        out = self.point_to_cell(pts, feats, self.cell_W, self.cell_H, self.cell_D)
        return out

    def point_to_cell(self, pts, feat, W, H, D, expand=1):
        """ perform maxpool on points in every cell set zero vector if cell is
        empty if expand=1 then return (N+1)x(N+1)x(N+1), for dmc xpand=0 then
        return NxNxN, for occupancy/sdf baselines
        """
        batchsize = feat.size()[0]
        C = feat.size()[2] 

        feat_cell = []
        # grid_shape = torch.LongTensor([W, H, D])
        for k in range(batchsize):
            feat_cell.append(self.grid_pool(feat[k, :, :], pts[k, :, :]))

        feat_cell = torch.stack(feat_cell, dim=0)

        # TODO check if this view is compatible to output of grid pool
        feat_cell = torch.transpose(feat_cell, 1, 2).contiguous().view(
            -1, C, W, H, D)
        if expand == 0:
            return feat_cell

        # expand to (W+1)x(H+1)
        curr_size = feat_cell.size()
        feat_cell_exp = torch.zeros(
            curr_size[0], curr_size[1], curr_size[2]+1, curr_size[3]+1,
            curr_size[4]+1).to(pts.device)
        feat_cell_exp[:, :, :-1, :-1, :-1] = feat_cell
        return feat_cell_exp
