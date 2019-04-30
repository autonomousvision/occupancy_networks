import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
#import settings
from im2mesh.dmc.utils.util import (
    offset_to_normal, offset_to_vertices, pts_in_cell, dis_to_meshs)
from im2mesh.dmc.ops.table import (
    get_connected_pairs, get_accept_topology)
import scipy.ndimage


class LossAutoGrad(object):
    """Implement the loss functions using pytorch,
       used in cffi/test/ for gradient checking
    """
    def __init__(self, num_cells, len_cell):

        self.len_cell = len_cell
        self.num_cells = num_cells
        self.dtype = torch.cuda.FloatTensor
        self.dtype_long = torch.cuda.LongTensor

        self.x_grids = np.arange(0, num_cells+1, len_cell)
        self.y_grids = np.arange(0, num_cells+1, len_cell)
        self.z_grids = np.arange(0, num_cells+1, len_cell)

        self.xv_value, self.yv_value, self.zv_value = np.meshgrid(self.x_grids[:-1], self.y_grids[:-1], self.z_grids[:-1], indexing='ij')
        self.xv_value = self.xv_value.flatten()
        self.yv_value = self.yv_value.flatten()
        self.zv_value = self.zv_value.flatten()

        connected_x, connected_y, connected_z, connected_inner, topology_to_triangle = get_connected_pairs()
        self.nonzero_connection = np.sum(connected_x) + np.sum(connected_y) + np.sum(connected_z)
        self.connected_x = connected_x
        self.connected_y = connected_y
        self.connected_z = connected_z
        self.connected_inner = connected_inner
        self.topology_to_triangle = topology_to_triangle

        self.acceptTopology = torch.LongTensor(get_accept_topology())
        self.acceptTopology = self.acceptTopology.cuda()
        flip_indices = torch.arange(self.acceptTopology.size()[0]-1, -1, -1).type(self.dtype_long)
        self.acceptTopologyWithFlip = torch.cat([self.acceptTopology, 255-self.acceptTopology[flip_indices]], dim=0)

        # note we consider the topology with 4 triangles only for visualizing
        # will be fixed in the future
        self.visTopology = torch.LongTensor(get_accept_topology(4))
        self.visTopology = self.visTopology.cuda()



    def loss_point_to_mesh_distance_autograd(self, offset, point, phase='train'):
        """ Compute the point-to-mesh distance using pytorch,
            implemented for gradient check of the c/c++ extensions
        """

        dis_empty = Variable(torch.ones(48).type(self.dtype) * 0.4)
        dis_empty[-1].item = 0.0

        distance_auto = []
        for i_,(x_,y_,z_) in enumerate(zip(self.xv_value, self.yv_value, self.zv_value)): 

            pts_cell = pts_in_cell(torch.unsqueeze(point, 0), [x_,y_,z_,
                x_+self.len_cell,y_+self.len_cell,z_+self.len_cell])
            if len(pts_cell)==0:
                dis = dis_empty
                mdis, mind = torch.min(dis, 0)
                mind = mind.item

            else:
                vertices = offset_to_vertices(offset,
                                              np.where(self.x_grids == x_)[0][0],
                                              np.where(self.y_grids == y_)[0][0],
                                              np.where(self.z_grids == z_)[0][0])
                dis = dis_to_meshs(torch.unsqueeze(point, 0), pts_cell, vertices, x_, y_, z_)
            distance_auto.append(torch.unsqueeze(dis, 1))

        distance_auto = torch.t(torch.cat(distance_auto, dim=1))
        return distance_auto


    def loss_on_curvature_autograd(self, offset, topology):
        """ Compute the curvature loss using pytorch,
            implemented for gradient check of the c/c++ extensions
        """
        loss = 0

        connected_x = self.connected_x
        connected_y = self.connected_y
        connected_z = self.connected_z
        connected_inner = self.connected_inner
        topology_to_triangle = self.topology_to_triangle
        for i_,(x_,y_,z_) in enumerate(zip(self.xv_value, self.yv_value, self.zv_value)): 
            # x direction
            if x_ != self.x_grids[-2]:
		# similarity constraint matrix
                # create new Variable from the data to avoid gradients on the topology
                # as the topology is only taken as a constant weight matrix 
                p1 = Variable((F.softmax(topology[i_,:], dim=0).data).type(self.dtype), requires_grad=True)
                p2 = Variable((F.softmax(topology[i_+self.num_cells*self.num_cells,:], dim=0).data).type(self.dtype), requires_grad=True)

                # expand the topology probability to triangle probability
                p1 = p1[torch.LongTensor(topology_to_triangle).type(self.dtype_long)]
                p2 = p2[torch.LongTensor(topology_to_triangle).type(self.dtype_long)]

                p_outer = torch.ger(p1, p2)
                W = torch.mul(p_outer, Variable(self.dtype(connected_x)))
                D1 = torch.diag(torch.sum(W, dim=1).view(-1))
                D2 = torch.diag(torch.sum(W, dim=0).view(-1))

		# get normal vector of triangles
                norm1 = offset_to_normal(offset, np.where(self.x_grids==x_)[0][0], np.where(self.y_grids==y_)[0][0], np.where(self.z_grids==z_)[0][0], 0)
                norm2 = offset_to_normal(offset, np.where(self.x_grids==x_+1)[0][0], np.where(self.y_grids==y_)[0][0], np.where(self.z_grids==z_)[0][0], 1)

                # normalize normal vectors
                norm1 = torch.div(norm1, torch.norm(norm1, 2, 1).unsqueeze(1).expand_as(norm1))
                norm2 = torch.div(norm2, torch.norm(norm2, 2, 1).unsqueeze(1).expand_as(norm2))

                # loss from matrix  
                tmp3 = torch.mm(torch.mm(norm1.transpose(0,1), W ), norm2)
                loss1 = torch.sum(W)*2 - torch.trace(tmp3)*2 
                loss += loss1

            # y direction
            if y_ != self.y_grids[-2]:
		# similarity constraint matrix
                # create new Variable from the data to avoid gradients on the topology
                # as the topology is only taken as a constant weight matrix 
                p1 = Variable((F.softmax(topology[i_,:], dim=0).data).type(self.dtype), requires_grad=True)
                p2 = Variable((F.softmax(topology[i_+self.num_cells,:], dim=0).data).type(self.dtype), requires_grad=True)

                # expand the topology probability to triangle probability
                p1 = p1[torch.LongTensor(topology_to_triangle).type(self.dtype_long)]
                p2 = p2[torch.LongTensor(topology_to_triangle).type(self.dtype_long)]

                p_outer = torch.ger(p1, p2)
                W = torch.mul(p_outer, Variable(self.dtype(connected_y)))
                D1 = torch.diag(torch.sum(W, dim=1).view(-1))
                D2 = torch.diag(torch.sum(W, dim=0).view(-1))

		# get normal vector of triangles
                norm1 = offset_to_normal(offset, np.where(self.x_grids==x_)[0][0], np.where(self.y_grids==y_)[0][0], np.where(self.z_grids==z_)[0][0], 2)
                norm2 = offset_to_normal(offset, np.where(self.x_grids==x_)[0][0], np.where(self.y_grids==y_+1)[0][0], np.where(self.z_grids==z_)[0][0], 3)

                # normalize normal vectors
                norm1 = torch.div(norm1, torch.norm(norm1, 2, 1).unsqueeze(1).expand_as(norm1))
                norm2 = torch.div(norm2, torch.norm(norm2, 2, 1).unsqueeze(1).expand_as(norm2))

                # loss from matrix  
                tmp3 = torch.mm(torch.mm(norm1.transpose(0,1), W ), norm2)
                loss1 = torch.sum(W)*2 - torch.trace(tmp3)*2 
                loss += loss1

            # z direction
            if z_ != self.z_grids[-2]:
		# similarity constraint matrix
                # create new Variable from the data to avoid gradients on the topology
                # as the topology is only taken as a constant weight matrix 
                p1 = Variable((F.softmax(topology[i_,:], dim=0).data).type(self.dtype), requires_grad=True)
                p2 = Variable((F.softmax(topology[i_+1,:], dim=0).data).type(self.dtype), requires_grad=True)

                # expand the topology probability to triangle probability
                p1 = p1[torch.LongTensor(topology_to_triangle).type(self.dtype_long)]
                p2 = p2[torch.LongTensor(topology_to_triangle).type(self.dtype_long)] 
                p_outer = torch.ger(p1, p2)
                W = torch.mul(p_outer, Variable(self.dtype(connected_z)))
                D1 = torch.diag(torch.sum(W, dim=1).view(-1))
                D2 = torch.diag(torch.sum(W, dim=0).view(-1))

		# get normal vector of triangles
                norm1 = offset_to_normal(offset, np.where(self.x_grids==x_)[0][0], np.where(self.y_grids==y_)[0][0], np.where(self.z_grids==z_)[0][0], 4)
                norm2 = offset_to_normal(offset, np.where(self.x_grids==x_)[0][0], np.where(self.y_grids==y_)[0][0], np.where(self.z_grids==z_+1)[0][0], 5)

                # normalize normal vectors
                norm1 = torch.div(norm1, torch.norm(norm1, 2, 1).unsqueeze(1).expand_as(norm1))
                norm2 = torch.div(norm2, torch.norm(norm2, 2, 1).unsqueeze(1).expand_as(norm2))

                # loss from matrix  
                tmp3 = torch.mm(torch.mm(norm1.transpose(0,1), W ), norm2)
                loss1 = torch.sum(W)*2 - torch.trace(tmp3)*2 
                loss += loss1

            # inner cell 
	    # similarity constraint matrix
            # create new Variable from the data to avoid gradients on the topology
            # as the topology is only taken as a constant weight matrix 
            p1 = Variable((F.softmax(topology[i_,:], dim=0).data).type(self.dtype), requires_grad=True)

            # expand the topology probability to triangle probability
            p1 = p1[torch.LongTensor(topology_to_triangle).type(self.dtype_long)]
            p_outer = torch.ger(p1, p1)
            W = torch.mul(p_outer, Variable(self.dtype(connected_inner)))
            D1 = torch.diag(torch.sum(W, dim=1).view(-1))
            D2 = torch.diag(torch.sum(W, dim=0).view(-1))

	    # get normal vector of triangles
            norm1 = offset_to_normal(offset, np.where(self.x_grids==x_)[0][0], np.where(self.y_grids==y_)[0][0], np.where(self.z_grids==z_)[0][0], 6)

            # normalize normal vectors
            norm1 = torch.div(norm1, torch.norm(norm1, 2, 1).unsqueeze(1).expand_as(norm1))

            # loss from matrix  
            tmp3 = torch.mm(torch.mm(norm1.transpose(0,1), W ), norm1)
            loss1 = torch.sum(W)*2 - torch.trace(tmp3)*2 
            loss += loss1
        return loss


    def loss_on_smoothness_autograd(self, occupancy):
        """ Compute the smoothness loss using pytorch,
            implemented for gradient check of the c/c++ extensions
        """

        W=occupancy.size()[0]
        H=occupancy.size()[1]
        D=occupancy.size()[2]

        loss = 0
        for x_ in range(W):
            for y_ in range(H):
                for z_ in range(D):
                    # horizontal direction
                    if x_<W-1:
                        # l1 loss
                        loss += torch.abs(occupancy[x_, y_, z_]-occupancy[x_+1,y_,z_])
                    # vertical direction
                    if y_<H-1:
                        # l1 loss
                        loss += torch.abs(occupancy[x_, y_, z_]-occupancy[x_,y_+1,z_])
                    if z_<D-1:
                        # l1 loss
                        loss += torch.abs(occupancy[x_, y_, z_]-occupancy[x_,y_,z_+1])

        return loss


