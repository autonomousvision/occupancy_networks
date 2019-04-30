
import sys
sys.path.append('../../../..')

import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import numpy as np
import resource

from im2mesh.dmc.ops.tests.loss_autograd import LossAutoGrad
from im2mesh.dmc.ops.point_triangle_distance import PointTriangleDistance


print("Testing CUDA extension...")
dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor
num_cells = 4
# autograd loss
loss_autograd = LossAutoGrad(num_cells, 1.0)

multiGrids = PointTriangleDistance()


if __name__ == '__main__':

    print("=========== Input =============")    
    point = Variable(torch.rand(10, 3).view(-1,3).type(dtype) * 0.9) * num_cells 
    offset = Variable(torch.zeros(3, num_cells+1, num_cells+1, num_cells+1).type(dtype)*0.5, requires_grad=True)
    print(point.shape)
    print(offset.shape)
    
    print("============= cuda extension ============")
    # forward
    tf_c = time.time()
    distance = multiGrids.forward(offset, point)
    tf_c = time.time() - tf_c
    distance_np = distance.data.cpu().numpy()
    print("cffi distance:")
    print(distance_np.shape)

    weight_rnd = Variable(torch.rand(distance.size()).type(dtype), requires_grad=False)
    distance_sum = torch.sum(torch.mul(distance, weight_rnd))
    
    # backward
    tb_c = time.time()
    grad = distance_sum.backward()
    tb_c = time.time() - tb_c
    offset_np = np.copy(offset.grad.data.cpu().numpy())
    
    print("cffi grad:")
    print(offset_np.shape)
    
    print("============= auto ============")
    # forward
    tf_py = time.time()
    distance_auto = loss_autograd.loss_point_to_mesh_distance_autograd(offset, point)
    tf_py = time.time()-tf_py
    distance_auto_np = distance_auto.data.cpu().numpy()
    print("auto distance:")
    print(distance_auto_np.shape)
    weight_rnd = Variable(weight_rnd.data)
    distance_sum_auto = torch.sum(torch.mul(distance_auto, weight_rnd))

    # backward
    offset.grad.data.zero_()
    
    tb_py = time.time()
    distance_sum_auto.backward()
    tb_py = time.time() - tb_py
    print("auto grad: ")
    offset_auto_np = np.copy(offset.grad.data.cpu().numpy())
    print(offset_auto_np.shape)

    print("========== summary ===========")
    print("Forward difference between cffi and auto: "+str(np.sum(np.abs(distance_np[:,:-1]-distance_auto_np[:,:-1]))))
    print("Backward difference between cffi and auto: "+str(np.sum(np.abs(offset_np-offset_auto_np))))