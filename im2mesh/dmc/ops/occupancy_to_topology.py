import math
from torch import nn
from torch.autograd import Function
import torch
from ._cuda_ext import occupancy_to_topology_forward, occupancy_to_topology_backward



class OccupancyToTopologyFunction(Function):
    @staticmethod
    def forward(ctx, occupancy):
        W = occupancy.size()[0] - 1
        H = occupancy.size()[1] - 1
        D = occupancy.size()[2] - 1

        T = 256
        topology = torch.zeros((W*H*D, T), dtype=torch.float32, device='cuda')
        occupancy_to_topology_forward(occupancy, topology)

        ctx.save_for_backward(occupancy, topology)

        return topology 

    @staticmethod
    def backward(ctx, grad_output):
        occupancy, topology = ctx.saved_tensors
        grad_occupancy = torch.zeros(occupancy.size(), dtype=torch.float32, device='cuda')
        occupancy_to_topology_backward(grad_output, occupancy, topology, grad_occupancy)
        return grad_occupancy


class OccupancyToTopology(nn.Module):
    """
    Module for deriving the topology probabilities of each cell given the occupancy probabilities

    Init
    ----------
    args1: shape of the topology output [W*H*DxT]
    
    Forward
    ----------
    arg1 : tensor
        occupancy probability tensor [W+1xH+1xD+1]

    Returns
    -------
    tensor
        topology probability tensor [W*H*DxT]

    """
    def __init__(self):
        super(OccupancyToTopology, self).__init__()
    def forward(self, occupancy):
        return OccupancyToTopologyFunction.apply(occupancy)  
