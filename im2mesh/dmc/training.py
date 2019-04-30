import os
from tqdm import trange
import torch
from im2mesh.common import chamfer_distance
from im2mesh.training import BaseTrainer
from im2mesh.utils import visualize as vis
import numpy as np
import torch.nn.functional as F
import scipy.ndimage

from im2mesh.dmc.utils.util import gaussian_kernel, offset_to_normal
from im2mesh.dmc.ops.curvature_constraint import CurvatureConstraint
from im2mesh.dmc.ops.occupancy_connectivity import OccupancyConnectivity
from im2mesh.dmc.ops.point_triangle_distance import PointTriangleDistance
from im2mesh.dmc.ops.table import get_accept_topology


class Trainer(BaseTrainer):
    def __init__(self, model, optimizer, device=None, input_type='pointcloud',
                 vis_dir=None, num_voxels=16, weight_distance=5.0, weight_prior_pos=0.2, weight_prior=10.0, weight_smoothness=3.0, weight_curvature=3.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        self.num_cells = num_voxels # - 1
        self.len_cell = 1.0

        self.x_grids = np.arange(0, self.num_cells+1, self.len_cell)
        self.y_grids = np.arange(0, self.num_cells+1, self.len_cell)
        self.z_grids = np.arange(0, self.num_cells+1, self.len_cell)
        self.distanceModule = PointTriangleDistance()
        self.curvatureLoss = CurvatureConstraint()
        self.occupancyConnectivity = OccupancyConnectivity()

        self.acceptTopology = torch.LongTensor(
            get_accept_topology()).to(device)
        flip_indices = torch.arange(
            self.acceptTopology.size()[0]-1, -1, -1).long()
        self.acceptTopologyWithFlip = torch.cat([
            self.acceptTopology, 255 - self.acceptTopology[flip_indices]], dim=0)

        self.visTopology = torch.LongTensor(get_accept_topology(4)).to(device)

        # assume that the outside __faces__ of the grid is always free
        W = len(self.x_grids)
        H = len(self.y_grids)
        D = len(self.z_grids)

        tmp_ = np.zeros((W, H, D))
        tmp_[0, :, :] = 1
        tmp_[W-1, :, :] = 1
        tmp_[:, :, 0] = 1
        tmp_[:, :, D-1] = 1
        tmp_[:, 0, :] = 1
        tmp_[:, H-1, :] = 1
        kern3 = gaussian_kernel(3)
        neg_weight = scipy.ndimage.filters.convolve(tmp_, kern3)
        neg_weight = neg_weight/np.max(neg_weight)
        self.neg_weight = torch.from_numpy(neg_weight.astype(np.float32)).to(device)
        self.one = torch.ones(1, requires_grad=True).to(device)
        self.weight_distance = weight_distance
        self.weight_prior_pos = weight_prior_pos
        self.weight_prior = weight_prior
        self.weight_smoothness = weight_smoothness
        self.weight_curvature = weight_curvature

                
    def train_step(self, data):
        self.model.train()

        inputs = data.get('inputs').to(self.device)
        pointcloud = data.get('pointcloud').to(self.device)

        inputs = self.num_cells * (inputs / 1.2 + 0.5)
        pointcloud = self.num_cells * (pointcloud / 1.2 + 0.5)

        offset, topology, occupancy = self.model(inputs)
        
        loss, loss_stages = self.loss_train(
            offset, topology, pointcloud, occupancy)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        self.model.eval()
        device = self.device

        inputs = data.get('inputs').to(device)
        pointcloud = data.get('pointcloud').to(device)

        inputs = self.num_cells * (inputs / 1.2 + 0.5)
        pointcloud = self.num_cells * (pointcloud / 1.2 + 0.5)

        with torch.no_grad():
            offset, topology, occupancy = self.model(inputs)
            loss, loss_stages = self.loss_train(
                offset, topology, pointcloud, occupancy)

        loss = loss.item()

        eval_dict = {
            'loss': loss,
            'loss mesh': loss_stages[0],
            'loss occupancy': loss_stages[1],
            'loss smoothness': loss_stages[2],
            'loss curvature': loss_stages[3],
        }

        return eval_dict

    def visualize(self, data):
        device = self.device
        shape = (self.num_cells + 1,) * 3
        inputs = data.get('inputs').to(self.device)
        batch_size = inputs.size(0)

        inputs_norm = self.num_cells * (inputs / 1.2 + 0.5)

        with torch.no_grad():
            offset, topology, occupancy = self.model(inputs_norm)
        
        occupancy = occupancy.view(batch_size, *shape)
        voxels_out = (occupancy >= 0.5).cpu().numpy()

        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))



    def loss_train(self, offset, topology, pts, occupancy):
        """Compute the losses given a batch of point cloud and the predicted
        mesh during the training phase
        """
        loss = 0
        loss_stages = []
        batchsize = offset.size()[0]

        for i in range(batchsize):
            # L^{mesh}
            loss += self.loss_point_to_mesh(offset[i], topology[i], pts[i], 'train')
            if i == 0:
                loss_stages.append(loss.item())

            # L^{occ}
            loss += self.loss_on_occupancy(occupancy[i, 0])
            if i == 0:
                loss_stages.append(loss.item() - sum(loss_stages))

            # L^{smooth}
            loss += self.loss_on_smoothness(occupancy[i, 0])
            if i == 0:
                loss_stages.append(loss.item() - sum(loss_stages))

            # L^{curve}
            loss += self.loss_on_curvature(offset[i], topology[i])
            if i == 0:
                loss_stages.append(loss.item() - sum(loss_stages))

        loss = loss/batchsize

        return loss, loss_stages
  
  
    def loss_eval(self, offset, topology, pts):
        """Compute the point to mesh loss during validation phase"""
        loss = self.loss_point_to_mesh(offset, topology, pts, 'val')
        return loss * self.one

    def loss_point_to_mesh(self, offset, topology, pts, phase='train'):
        """Compute the point to mesh distance"""

        # compute the distances between all topologies and a point set
        dis_sub = self.distanceModule(offset, pts)

        # dual topologies share the same point-to-triangle distance
        flip_indices = torch.arange(len(self.acceptTopology)-1, -1, -1).long()
        dis_accepted = torch.cat([dis_sub, dis_sub[:, flip_indices]], dim=1)
        topology_accepted = topology[:, self.acceptTopologyWithFlip]

        # renormalize all desired topologies so that they sum to 1
        prob_sum = torch.sum(
            topology_accepted, dim=1, keepdim=True).clamp(1e-6)
        topology_accepted = topology_accepted / prob_sum

        # compute the expected loss
        loss = torch.sum(
            topology_accepted.mul(dis_accepted)) / (self.num_cells**3)

        if phase == 'train':
            loss = loss * self.weight_distance
        return loss

    def loss_on_occupancy(self, occupancy):
        """Compute the loss given the prior that the 6 faces of the cube 
        bounding the 3D scene are unoccupied and a sub-volume inside thec
        scene is occupied
        """
        # loss on 6 faces of the cube
        loss_free = torch.sum(torch.mul(
            occupancy, self.neg_weight)) / torch.sum(self.neg_weight)

        W = occupancy.size()[0]
        H = occupancy.size()[1]
        D = occupancy.size()[2]

        # get occupancy.data as we don't want to backpropagate to the adaptive_weight
        sorted_cube, _ = torch.sort(
            occupancy.data.view(-1), 0, descending=True)
        # check the largest 1/30 value
        adaptive_weight = 1 - torch.mean(sorted_cube[0:int(sorted_cube.size()[0]/30)])

        # loss on a subvolume inside the cube, where the weight is assigned
        # adaptively w.r.t. the current occupancy status 
        loss_occupied = self.weight_prior_pos * adaptive_weight * \
                (1-torch.mean(occupancy[int(0.2*W):int(0.8*W), 
                int(0.2*H):int(0.8*H), int(0.2*D):int(0.8*D)]))

        return (loss_free + loss_occupied) * self.weight_prior

    def loss_on_smoothness(self, occupancy):
        """Compute the smoothness loss defined between neighboring occupancy
        variables
        """
        loss = (
            self.occupancyConnectivity(occupancy) / (self.num_cells**3)
            * self.weight_smoothness
        )
        return  loss

    def loss_on_curvature(self, offset, topology):
        """Compute the curvature loss by measuring the smoothness of the
        predicted mesh geometry
        """
        topology_accepted = topology[:, self.acceptTopologyWithFlip]
        return self.weight_curvature*self.curvatureLoss(offset, \
                F.softmax(topology_accepted, dim=1)) / (self.num_cells**3)


