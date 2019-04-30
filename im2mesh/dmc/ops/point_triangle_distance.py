import torch
import math
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
from im2mesh.dmc.ops.table import get_connected_pairs
from ._cuda_ext import point_topology_distance_forward, point_topology_distance_backward


class PointTriangleDistanceFunction(Function):
    @staticmethod
    def forward(ctx, offset, points):
        W = offset.size()[1]
        H = offset.size()[2]
        D = offset.size()[3]

        # we only considered topologies with up to 3 triangles for calculating
        # the distance loss function, the distance can be calculated in regardless
        # of the normal vectors, therefore there are only 48 topologies to be
        # considered
        T = 48

        distances_full = torch.zeros((W-1)*(H-1)*(D-1), T).cuda()
        indices = -1 * torch.ones((points.size(0), T), dtype=torch.int32, device='cuda')
        point_topology_distance_forward(
                offset, points, distances_full, indices) 
        ctx.save_for_backward(offset, points, indices)
        return distances_full 

    @staticmethod
    def backward(ctx, grad_output):
        offset, points, indices = ctx.saved_tensors

        grad_offset = torch.zeros(offset.size(), device='cuda')
        point_topology_distance_backward(
                grad_output, offset, points, indices, grad_offset)
        return grad_offset, None 


class PointTriangleDistance(nn.Module):

    """
    Module for deriving the Point to Triangle distance 
    (for each topology with up to 3 triangles)

    Forward
    ----------
    arg1 : tensor
        offset variable [3 x W+1 x H+1 x D+1]
    
    arg1 : tensor
        points [N x 3]

    Returns
    -------
    tensor
         distance [W*H*D x T]

    """

    def __init__(self):
        super(PointTriangleDistance, self).__init__()
    def forward(self, offset, points):
        return PointTriangleDistanceFunction.apply(offset, points)  
