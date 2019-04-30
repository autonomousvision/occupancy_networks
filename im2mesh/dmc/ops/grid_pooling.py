import torch
import math
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
from ._cuda_ext import grid_pooling_forward, grid_pooling_backward


class GridPoolingFunction(Function):
    """ Perform max-pooling in every cell over the point features
        see ../src/extension.cpp
            ../src/grid_pooling_kernel.cu
        for more details
    """
    @staticmethod
    def forward(ctx, feat_points, points, grid_shape):
        feat_points = feat_points.contiguous()
        points = points.contiguous()
        W = grid_shape[0]
        H = grid_shape[1]
        D = grid_shape[2]
        C = feat_points.size()[1]
        grid_shape = grid_shape.cpu().contiguous()
        feat_cells = torch.zeros((W*H*D, C), dtype=torch.float32, device='cuda')
        indices = -1 * torch.ones((W*H*D, C), dtype=torch.int32, device='cuda')
        grid_pooling_forward(points, feat_points, grid_shape, feat_cells, indices)

        # save for back-propagation
        ctx.save_for_backward(indices, grid_shape)
        # save number of points and feature dimension for back-propagation
        ctx.N = points.size()[0]
        ctx.C = C

        return feat_cells

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        indices, grid_shape = ctx.saved_tensors
        N, C = ctx.N, ctx.C
        grad_points = torch.zeros((N, C), dtype=torch.float32, device='cuda')
        grid_pooling_backward(grad_output, grid_shape, indices, grad_points)
        # we only need gradient on feat_points
        return grad_points, None, None


class GridPooling(nn.Module):

    """
    Module for Grid Pooling from Points with features to gird cells with features

    Init
    ----------
    args1: gridshape [3]

    
    Forward
    ----------
    arg1 : tensor
        point features [N x F]
    
    arg1 : tensor
        point locations [N x 3]

    Returns
    -------
    tensor
        Feature grid [W*H*D x F]

    """

    def __init__(self, gridshape):
        super(GridPooling, self).__init__()
        self.gridshape = gridshape

    def forward(self, features, points):
        return GridPoolingFunction.apply(features, points, self.gridshape)  
