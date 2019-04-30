import sys
sys.path.append('../../../..')
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import torch.nn.functional as F
from im2mesh.dmc.ops.occupancy_to_topology import OccupancyToTopology

def get_occupancy_table():
    """Return binary occupancy status of 8 vertices for all 256 topology types"""
    occTable = np.zeros((256, 8)) 
    for x in range(256):
        for v in range(8):
            occTable[x, v] = int(x)&(pow(2,v))!=0 

    return occTable

# look-up-tables
acceptTopology = np.arange(256)
vertexTable=[ [0, 1, 0],
	      [1, 1, 0],
	      [1, 0, 0],
              [0, 0, 0],
	      [0, 1, 1],
	      [1, 1, 1],
	      [1, 0, 1],
              [0, 0, 1] ]
occupancyTable=get_occupancy_table()

# check the cuda extension or c extension


dtype_gpu = torch.cuda.FloatTensor
dtype_cpu = torch.FloatTensor


# get (WxH)xT topology map from (W+1)x(Hx1) occupancy map
# note here T=14 because of the inside/outside distinction
def occupancy_to_topology(occ):
    Wc = occ.size()[0]-1
    Hc = occ.size()[1]-1
    Dc = occ.size()[2]-1
    T = len(acceptTopology)
    topology = Variable(torch.zeros(Wc*Hc*Dc, T)).type(torch.FloatTensor)
    #vertexTablee = torch.from_numpy(np.array(vertexTable)).cuda()

    xv, yv, zv = np.meshgrid(range(Wc), range(Hc), range(Dc), indexing='ij')
    xv = xv.flatten()
    yv = yv.flatten()
    zv = zv.flatten()
    
    for i,j,k in zip(xv, yv, zv):
        p_occ = [] 
        for v in range(8):
            p_occ.append( occ[i+vertexTable[v][0], j+vertexTable[v][1], k+vertexTable[v][2]] )
            p_occ.append( 1 - occ[i+vertexTable[v][0], j+vertexTable[v][1], k+vertexTable[v][2]] )
        for t in range(T):
            topology_ind = acceptTopology[t]
            p_accumu = 1
            for v in range(8):
                p_accumu = p_accumu*p_occ[ v*2 + int(occupancyTable[topology_ind][v]) ] 
            topology[i*H*D+j*D+k, t] = p_accumu
    return topology



if __name__ == '__main__':

    W = H = D = 4
    T = 256

    print("=========== Input =============")
    
    occupancy = Variable(torch.rand(W+1, H+1, D+1).type(dtype_cpu), requires_grad=True)
    rnd_weights = Variable(torch.rand(W*H*D, T).type(dtype_cpu))
    print("Occupancy shape: "+str(occupancy.shape))

    print("============= Normal Pytorch ============")
    
    # forward
    tf_c = time.time()
    topo = occupancy_to_topology(occupancy)
    tf_c_ = (time.time() - tf_c)*1000 
    print("normal forward time: {:.2} ms".format(tf_c_))

    # backward
    tb_py = time.time()
    torch.sum(torch.mul(topo, rnd_weights)).backward()
    tb_py = (time.time()-tb_py)*1000
    print("auto backward time: {:.2} ms".format(tb_py))
    grad_auto_np = np.copy(occupancy.grad.data.cpu().numpy())
    print(grad_auto_np)

    print("============= Cuda Extension ============")
    #occupancy.grad.data.zero_()
    occupancy2 = Variable(occupancy.data.cuda(), requires_grad=True)
    #forward
    #occ2topo_modul = OccupancyToTopology()
    tf_c = time.time()
    topology = OccupancyToTopology()(occupancy2)# occ2topo_modul.forward(occupancy)
    tf_cf = (time.time() - tf_c)*1000
    print("Cuda forward time: {:3.2} ms".format(tf_cf))
    
    # backward
    tb_c = time.time()
    loss = torch.sum(torch.mul(topology, rnd_weights.cuda()))
    loss.backward()
    tb_cb = (time.time() - tb_c)*1000
    print("Cuda backward time: {:3.2} ms".format(tb_cb))
    grad_np = np.copy(occupancy2.grad.data.cpu().numpy())
    print(grad_np)
    print("============= Comparison Forward ============")

    print("Topolgy shape: "+str(topology.shape))
    print("Forward sum L1 pytroch vs cuda: {:.2} ".format(np.sum(np.abs(topology.data.cpu().numpy()-topo.data.cpu().numpy()))))
    print("Backward sum L1 pytroch vs cuda: {:.2} ".format(np.sum(np.abs(grad_auto_np - grad_np))))
    print("Forward cuda extension is {:.0} times faster".format((tf_c_/tf_cf)))
    print("Backward cuda extension is {:.0} times faster".format((tb_py/tb_cb)))