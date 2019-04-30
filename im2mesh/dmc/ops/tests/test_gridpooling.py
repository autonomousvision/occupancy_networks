import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import time
import numpy as np
import sys

sys.path.append('../../../..')
from im2mesh.dmc.utils.util import pts_in_cell 
from torch.autograd import gradcheck

from im2mesh.dmc.ops.grid_pooling import GridPoolingFunction


np.set_printoptions(threshold='nan')

# check the cuda extension or c extension
print( "Testing CUDA extension...")
dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor



#W = 15 
#H = 15 
#D = 15 
N = 100
C = 8

num_cells = 4 
len_cell = 1.0
W = H = D = num_cells

x_grids = np.arange(0, num_cells+1, len_cell)
y_grids = np.arange(0, num_cells+1, len_cell)
z_grids = np.arange(0, num_cells+1, len_cell)

# perform maxpool on points in every cell
# set zero vector if cell is empty
def grid_pooling_auto(pts, feat):
    xv_value, yv_value, zv_value = np.meshgrid(x_grids[:-1], y_grids[:-1], z_grids[:-1], indexing='ij')
    xv_value = xv_value.flatten()
    yv_value = yv_value.flatten()
    zv_value = zv_value.flatten()

    
    feat_cell = Variable(torch.zeros((len(x_grids)-1) * (len(y_grids)-1) * (len(z_grids)-1), C).type(dtype))
    #for k in range(batchsize):
    for i_,(x_,y_,z_) in enumerate(zip(xv_value, yv_value, zv_value)): 
        pts_index = pts_in_cell(pts.unsqueeze(0),[x_,y_,z_,
            x_+len_cell, y_+len_cell, z_+len_cell])
        if len(pts_index)>0:
            pts_index = torch.LongTensor(pts_index).type(dtype_long)
            #pts_feat = feat.index_select(0, pts_index)
            pts_feat = feat[pts_index,:]
            # max pooling
            #pts_feat,_ = torch.max(pts_feat, 0)
            m = nn.MaxPool1d(len(pts_index))
            pts_feat = m(pts_feat.t().unsqueeze(0))
            feat_cell[i_, :] = pts_feat.squeeze()
    return feat_cell

#class GridPooling(Function):
#    def forward(self, points, feat_points):
#        feat_cells = torch.zeros(W*H*D, C).type(dtype)
#        indices = -1 * torch.ones(W*H*D, C).type(dtype_long)
#        shape = torch.LongTensor([W, H, D]).type(dtype_long)
#        forward_utils.grid_pooling_forward(points, feat_points, shape, feat_cells, indices) 
#        self.saved_indices = indices
#        return feat_cells 
#
#    def backward(self, grad_output):
#        grad_points = torch.zeros(N, C).type(torch.FloatTensor)
#        forward_utils.grid_pooling_backward( grad_output, self.saved_indices, grad_points) 
#        return None, grad_points 



if __name__ == '__main__':
    points = Variable(torch.rand(N, 3).view(-1,3).type(dtype), requires_grad=False) * 5.0
    feat_points = Variable(torch.rand(N, C).type(dtype), requires_grad=True)
    rnd_weights = Variable(torch.rand(W*H*D, C).type(dtype))
    shape = Variable(torch.LongTensor([W, H, D]))

    print( "=========== Input =============")
    print( points)
    print (feat_points)

    print( "============= cffi ============")
    # forward
    feat_cells = GridPoolingFunction.apply(feat_points, points, shape)
    tf_c = time.time()
    feat_cells = GridPoolingFunction.apply(feat_points, points, shape)
    tf_c = time.time() - tf_c
    print ("cffi forward time: ", tf_c
)
    # backward
    tb_c = time.time()
    torch.sum( torch.mul(feat_cells, rnd_weights) ).backward()
    tb_c = time.time() - tb_c
    print ("cffi backward time: ", tb_c)

    grad_np = np.copy(feat_points.grad.data.cpu().numpy())
    # print (grad_np)

    print( "============= auto ============")
    # forward
    tf_py = time.time()
    feat_cells_auto = grid_pooling_auto(points, feat_points)
    tf_py = time.time()-tf_py
    print ("auto forward time: ", tf_py)

    # backward
    feat_points.grad.data.zero_()
    tb_py = time.time()
    torch.sum(torch.mul(feat_cells_auto, rnd_weights)).backward()
    tb_py = time.time()-tb_py
    print( "auto backward time: ", tf_py)

    grad_auto_np = np.copy(feat_points.grad.data.cpu().numpy())
    # print (grad_auto_np)
    
    
    print ("========== summary ===========")
    print ("Forward difference between cffi and auto: ", np.sum(np.abs(feat_cells.data.cpu().numpy()-feat_cells_auto.data.cpu().numpy())))
    print ("Backward difference between cffi and auto: ", np.sum(np.abs(grad_np-grad_auto_np)))
    
    print ("cffi forward time: %f, backward time: %f, full time: %f " % (tf_c, tb_c, tf_c+tb_c))
    print ("auto forward time: %f, backward time: %f, full time: %f " % (tf_py, tb_py, tf_py+tb_py))
    print ("ratio: ", (tf_py+tb_py)/(tf_c + tb_c))
    
    