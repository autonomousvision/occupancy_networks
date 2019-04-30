import torch
import math
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
from ._cuda_ext import occupancy_connectivity_forward, occupancy_connectivity_backward



class OccupancyConnectivityFunction(Function):
    @staticmethod
    def forward(ctx, occ):
        loss = occupancy_connectivity_forward(occ)
        ctx.save_for_backward(occ)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        occ, = ctx.saved_tensors
        grad_occupancy = torch.zeros(occ.size(), dtype=torch.float32, device='cuda')
        occupancy_connectivity_backward(
            grad_output,
            occ,
            grad_occupancy)
        # Multiply with incoming gradient
        grad_occupancy = grad_occupancy * grad_output
        return grad_occupancy


class OccupancyConnectivity(nn.Module):

    """
    Module for deriving the Occupancy connectiviy loss 

    ForwardW
    ----------
    arg1 : tensor
        occupancy probabilities [W+1 x H+1 x D+1]

    Returns
    -------
    tensor
         Occupancy connectiviy loss 1

    """

    def __init__(self):
        super(OccupancyConnectivity, self).__init__()
    def forward(self, occ):
        return OccupancyConnectivityFunction.apply(occ)  
