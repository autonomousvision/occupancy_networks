import os
from tqdm import trange
import numpy as np
import torch
import torch.nn.functional as F
from im2mesh.training import BaseTrainer
from im2mesh.common import compute_iou
from im2mesh.utils import visualize as vis
from im2mesh.utils.voxels import VoxelGrid


class Trainer(BaseTrainer):
    ''' Trainer class for the R2N2 model.

    It handles the training and evaluation steps as well as intermidiate
    visualizations.

    Args:
        model (nn.Module): R2N2 model
        optimizer (optimizer): pytorch optimizer
        device (device): pytorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value
    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.5):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        occ = data.get('voxels').to(self.device)
        inputs = data.get('inputs').to(self.device)

        loss = self.compute_loss(occ, inputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        device = self.device
        threshold = self.threshold

        occ = data.get('voxels').to(device)
        inputs = data.get('inputs').to(device)
        points = data.get('points_iou')
        points_occ = data.get('points_iou.occ')

        with torch.no_grad():
            occ_logits = self.model(inputs).squeeze(1)

        eval_dict = {}

        # Compute loss
        occ_hat = torch.sigmoid(occ_logits)
        loss = F.binary_cross_entropy_with_logits(occ_logits, occ)
        eval_dict['loss'] = loss.item()

        # Compute discretized IOU
        occ_np = (occ >= 0.5).cpu().numpy()
        occ_hat_np = (occ_hat >= threshold).cpu().numpy()
        iou_voxels = compute_iou(occ_np, occ_hat_np).mean()
        eval_dict['iou_voxels'] = iou_voxels

        # Compute continuous IOU (if possible)
        if points is not None:
            voxel_grids = [VoxelGrid(occ_hat_np_i)
                           for occ_hat_np_i in occ_hat_np]
            points_np = points.cpu().numpy()
            points_occ_np = (points_occ >= 0.5).cpu().numpy()
            points_occ_hat_np = np.stack(
                [vg.contains(p) for p, vg in zip(points_np, voxel_grids)])
            iou = compute_iou(points_occ_np, points_occ_hat_np).mean()
            eval_dict['iou'] = iou

        return eval_dict

    def visualize(self, data):
        ''' Performs an intermidiate visualization.

        Args:
            data (dict): data dictionary
        '''
        device = self.device

        occ = data.get('voxels').to(device)
        inputs = data.get('inputs').to(device)

        with torch.no_grad():
            occ_logits = self.model(inputs).squeeze(1)

        occ_hat = torch.sigmoid(occ_logits)
        voxels_gt = (occ >= self.threshold).cpu().numpy()
        voxels_out = (occ_hat >= self.threshold).cpu().numpy()

        batch_size = occ.size(0)
        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            vis.visualize_voxels(
                voxels_out[i], os.path.join(self.vis_dir, '%03d.png' % i))
            vis.visualize_voxels(
                voxels_gt[i], os.path.join(self.vis_dir, '%03d_gt.png' % i))

    def compute_loss(self, occ, inputs=None):
        ''' Computes the loss.

        Args:
            occ (tensor): GT occupancy values for the voxel grid
            inputs (tensor): input tensor
        '''
        occ_hat = self.model(inputs).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(occ_hat, occ)
        return loss
