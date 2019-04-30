import torch
import math
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
from im2mesh.dmc.ops.table import get_connected_pairs
from ._cuda_ext import curvature_constraint_forward, curvature_constraint_backward


#########  TEST FAILS #########

# return connected pairs in x, y, z directions, inner cell pairs as well as a topolgy to triangles table
x, y, z, inner, topology_to_triangles = get_connected_pairs()


class CurvatureConstraintFunction(Function):
    @staticmethod
    def forward(ctx, offset, topology):
        loss = torch.zeros(1, dtype=torch.float32, device='cuda')
        loss = curvature_constraint_forward(
            offset,
            topology[:, torch.LongTensor(topology_to_triangles).cuda()],
            torch.FloatTensor(x).cuda(),
            torch.FloatTensor(y).cuda(),
            torch.FloatTensor(z).cuda(),
            torch.FloatTensor(inner).cuda())
        ctx.save_for_backward(offset, topology)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        offset, topology = ctx.saved_tensors

        grad_offset = torch.zeros(offset.size()).cuda()
        curvature_constraint_backward(
            grad_output,
            offset,
            topology[:, torch.LongTensor(topology_to_triangles).cuda()],
            torch.FloatTensor(x).cuda(),
            torch.FloatTensor(y).cuda(),
            torch.FloatTensor(z).cuda(),
            torch.FloatTensor(inner).cuda(),
            grad_offset)

        # Multiply with incoming gradient
        grad_offset = grad_offset * grad_output
        grad_topology = torch.zeros(topology.size()).cuda()
        return grad_offset, grad_topology 


class CurvatureConstraint(nn.Module):

    """
    #########  TEST FAILS #########
    Module for deriving the Curvature loss of each cell given the offset variables
    
    Forward
    ----------
    arg1 : tensor
        offset variables [3 x W+1 x H+1 x D+1]
    arg2 : tensor
        topology porbabilities [W*H*D x T]

    Returns
    -------
    tensor
        curvature loss 1

    """
    def __init__(self):
        super(CurvatureConstraint, self).__init__()
    def forward(self, off, topo):
        return CurvatureConstraintFunction.apply(off, topo)  
