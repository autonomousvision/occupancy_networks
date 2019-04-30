import os
from tqdm import trange
import torch
from im2mesh.common import chamfer_distance
from im2mesh.training import BaseTrainer
from im2mesh.utils import visualize as vis


class Trainer(BaseTrainer):
    r''' Trainer object for the Point Set Generation Network.

    The PSGN network is trained on Chamfer distance. The Trainer object
    obtains methods to perform a train and eval step as well as to visualize
    the current training state by plotting the respective point clouds.

    Args:
        model (nn.Module): PSGN model
        optiimzer (PyTorch optimizer): The optimizer that should be used
        device (PyTorch device): the PyTorch device
        input_type (string): The input type (e.g. 'img')
        vis_dir (string): the visualisation directory
    '''
    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        r''' Performs a train step.

        The chamfer loss is calculated and an appropriate backward pass is
        performed.

        Args:
            data (tensor): training data
        '''
        self.model.train()
        points = data.get('pointcloud').to(self.device)
        inputs = data.get('inputs').to(self.device)

        loss = self.compute_loss(points, inputs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        r''' Performs an evaluation step.

        The chamfer loss is calculated and returned in a dictionary.

        Args:
            data (tensor): input data
        '''
        self.model.eval()

        device = self.device

        points = data.get('pointcloud_chamfer').to(device)
        inputs = data.get('inputs').to(device)

        with torch.no_grad():
            points_out = self.model(inputs)

        loss = chamfer_distance(points, points_out).mean()
        loss = loss.item()
        eval_dict = {
            'loss': loss,
            'chamfer': loss,
        }

        return eval_dict

    def visualize(self, data):
        r''' Visualizes the current output data of the model.

        The point clouds for respective input data is plotted.

        Args:
            data (tensor): input data
        '''
        device = self.device

        points_gt = data.get('pointcloud').to(device)
        inputs = data.get('inputs').to(device)

        with torch.no_grad():
            points_out = self.model(inputs)

        points_out = points_out.cpu().numpy()
        points_gt = points_gt.cpu().numpy()

        batch_size = inputs.size(0)
        for i in trange(batch_size):
            input_img_path = os.path.join(self.vis_dir, '%03d_in.png' % i)
            vis.visualize_data(
                inputs[i].cpu(), self.input_type, input_img_path)
            out_file = os.path.join(self.vis_dir, '%03d.png' % i)
            out_file_gt = os.path.join(self.vis_dir, '%03d_gt.png' % i)
            vis.visualize_pointcloud(points_out[i], out_file=out_file)
            vis.visualize_pointcloud(points_gt[i], out_file=out_file_gt)

    def compute_loss(self, points, inputs):
        r''' Computes the loss.

        The Point Set Generation Network is trained on the Chamfer distance.

        Args:
            points (tensor): GT point cloud data
            inputs (tensor): input data for the model
        '''
        points_out = self.model(inputs)
        loss = chamfer_distance(points, points_out).mean()
        return loss
