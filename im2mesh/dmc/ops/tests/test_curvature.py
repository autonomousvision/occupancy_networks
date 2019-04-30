import torch
import torch.nn as nn
from torch.autograd import Variable
import sys

sys.path.append('../../../..')
from im2mesh.dmc.ops.tests.loss_autograd import LossAutoGrad
from im2mesh.dmc.ops.curvature_constraint import CurvatureConstraint 
import torch.nn.functional as F
import numpy as np
import time

# check the cuda extension or c extension

print ("Testing CUDA extension...")
dtype = torch.cuda.FloatTensor


# autograd loss
num_cells = 4 
len_cell = 1.0
W = H = D = num_cells

loss_autograd = LossAutoGrad(num_cells, len_cell)


# cffi loss
class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.smoothLoss = CurvatureConstraint()

    def forward(self, offset, topology):
        return self.smoothLoss(offset, topology)


if __name__ == '__main__':

    # generate offset and topology with relatively low-dimension
    print ("=========== Input =============")
    T = 96
    W = num_cells
    H = num_cells
    D = num_cells
    offset = Variable((torch.rand(3, W+1, H+1, D+1)).type(dtype) * 0.1, requires_grad=True)
    topology = Variable(torch.rand(W*H*D, T).type(dtype), requires_grad=True)
    #print (offset)
    #print (topology)

    loss_cffi = SmoothLoss()
    l = loss_cffi(offset, F.softmax(topology, dim=1))
    l.backward()
    offset.grad.data.zero_()

    # evaluating the running time of the cffi extension
    print ("============= cffi ============")
    tf_c = time.time()
    l = loss_cffi(offset, F.softmax(topology, dim=1))
    print ("cffi loss:")
    print (l)
    tf_c = time.time()-tf_c
    
    
    tb_c = time.time()
    l.backward()
    print ("cffi gradient:")
    print( offset.grad)
    tb_c = time.time()-tb_c
    grad_np = np.copy(offset.grad.data.cpu().numpy())
    
    
    # evaluating the running time of the autograd version 
    print ("============= auto ============")
    tf_py = time.time()
    l_auto = loss_autograd.loss_on_curvature_autograd(offset, topology)
    print ("auto loss:")
    print (l_auto)
    tf_py = time.time()-tf_py

    offset.grad.data.zero_()
    tb_py = time.time()
    l_auto.backward()
    print ("auto grad:")
    print (offset.grad)
    tb_py = time.time()-tb_py
    grad_auto_np = np.copy(offset.grad.data.cpu().numpy())
    assert np.sum(np.abs(grad_auto_np)) and np.sum(np.abs(grad_np)) != 0.0
    # print the loss and grad difference and the time comparison
    print ("========== summary ===========")
    print ("Forward difference between cffi and auto: ", (l-l_auto).data.cpu().numpy())
    print ("Backward difference between cffi and auto: ", np.sum(np.abs(grad_np-grad_auto_np)))
    print ("Backward difference between cffi and auto: ", np.mean(np.abs(grad_np-grad_auto_np)))

    print ("cffi forward time: %f, backward time: %f, full time: %f " % (tf_c, tb_c, tf_c+tb_c))
    print ("auto forward time: %f, backward time: %f, full time: %f " % (tf_py, tb_py, tf_py+tb_py))
    print ("ratio: ", (tf_py+tb_py)/(tf_c + tb_c))