
import sys
sys.path.append('../../../..')

import torch
import torch.nn as nn
from torch.autograd import Variable

import time
import numpy as np
from im2mesh.dmc.ops.occupancy_connectivity import OccupancyConnectivity
#from loss import Loss 
#from loss_autograd import LossAutoGrad 
#from parse_args import parse_args

# check the cuda extension or c extension

def loss_on_smoothness_autograd( occupancy):
    """ Compute the smoothness loss using pytorch,
        implemented for gradient check of the c/c++ extensions
    """

    Wo=occupancy.size()[0]
    Ho=occupancy.size()[1]
    Do=occupancy.size()[2]

    loss = 0
    for x_ in range(Wo):
        for y_ in range(Ho):
            for z_ in range(Do):
                # horizontal direction
                if x_<Wo-1:
                    # l1 loss
                    loss += torch.abs(occupancy[x_, y_, z_]-occupancy[x_+1,y_,z_])
                # vertical direction
                if y_<Ho-1:
                    # l1 loss
                    loss += torch.abs(occupancy[x_, y_, z_]-occupancy[x_,y_+1,z_])
                if z_<Do-1:
                    # l1 loss
                    loss += torch.abs(occupancy[x_, y_, z_]-occupancy[x_,y_,z_+1])

    return loss





W = H = D = 4

loss_mod = OccupancyConnectivity()
def loss_on_smoothness(occupancy):
    """Compute the smoothness loss defined between neighboring occupancy
    variables
    """ 
    return 1.0 *loss_mod.forward(occupancy)/ (W*H*D)

print("Testing CUDA extension...")
dtype = torch.cuda.FloatTensor


if __name__ == '__main__':

    occupancy = Variable(torch.rand(W+1, H+1, D+1).type(dtype), requires_grad=True)

    print("=========== Input =============")
    print(occupancy.shape)

    print("============= cffi ============")
    # forward
    tf_c = time.time()
    loss = loss_on_smoothness(occupancy)*(W*H*D)
    print(loss)
    tf_c = (time.time() - tf_c)*1000
    print("cffi forward time: {:.2} ms".format(tf_c))

    # backward
    tb_c = time.time()
    loss.backward()
    tb_c = (time.time() - tb_c)*1000
    print("cffi backward time:{:.2} ms".format(tb_c))
    grad_np = np.copy(occupancy.grad.data.cpu().numpy())
    print("gra mean"+str(np.mean(np.abs(grad_np))))

    print("============= auto ============")
    occupancy.grad.data.zero_()

    # forward
    tf_py = time.time()
    loss_auto =  loss_on_smoothness_autograd(occupancy) 
    tf_py = (time.time()-tf_py)*1000
    print("auto forward time:{:.2} ms".format(tf_py))
    print(loss_auto)

    # backward
    
    tb_py = time.time()
    loss_auto.backward()
    tb_py = (time.time()-tb_py)*1000
    print("auto backward time:{:.2} ms".format(tf_py))

    grad_auto_np = np.copy(occupancy.grad.data.cpu().numpy())
    assert np.sum(np.abs(grad_auto_np)) and np.sum(np.abs(grad_np)) != 0.0
    print("gra mean"+str(np.mean(np.abs(grad_auto_np))))

    print("Forward sum L1 pytroch vs cuda: {:.2} ".format(np.sum(np.abs(loss.detach().cpu().numpy()-loss_auto.detach().cpu().numpy()))))
    print("Backward sum L1 pytroch vs cuda: {:.2} ".format(np.sum(np.abs(grad_np-grad_auto_np))))
    

