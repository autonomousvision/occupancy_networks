import torch.nn.functional as F
import torch
from im2mesh.common import chamfer_distance
import os
from torchvision.utils import save_image
from im2mesh.training import BaseTrainer
from im2mesh.utils import visualize as vis
import im2mesh.common as common


class Trainer(BaseTrainer):
    r''' Trainer object for the pixel2mesh model.

    It provided methods to perform a training step, and evaluation step and
    necessary loss calculation functions. We adhered to the official
    Pixel2Mesh implementation where 4 different losses were used.

    Args:
        model (nn.Module): Pixel2Mesh module that should be trained
        optimizer (Optimizer): optimizer that should be used
        ellipsoid (numpy array): helper file with helper matrices for
                                 respective losses
        vis_dir (string): visualisation path
        device (device): The device that should be used (GPU or CPU)
    '''

    def __init__(
            self, model, optimizer, ellipsoid, vis_dir, device=None,
            adjust_losses=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.vis_dir = vis_dir
        # hardcoded indices and weights for the Laplace transformation
        self.lape_idx = ellipsoid[7]
        self.edges = []  # hardcoded IDs for edges in the mesh
        for i in range(1, 4):
            adj = ellipsoid[i][1]
            self.edges.append(torch.from_numpy(adj[0]).to(device))

        # Hyperparameters from the authors' implementation
        self.param_chamfer_w = 3000
        self.param_chamfer_rel = 0.55
        self.param_edge = 300
        self.param_n = 0.5
        self.param_lap = 1500
        self.param_lap_rel = 0.3
        self.param_move = 100

        if adjust_losses:
            print('Adjusting loss hyperparameters.')
            self.param_chamfer_w *= 0.57**2
            self.param_edge *= 0.57**2
            self.param_lap *= 0.57**2
            self.param_move *= 0.57**2

    def train_step(self, data):
        r''' Performs a training step of the model.

        Arguments:
            data (tensor): The input data
        '''

        self.model.train()
        points = data.get('pointcloud').to(self.device)
        normals = data.get('pointcloud.normals').to(self.device)
        img = data.get('inputs').to(self.device)
        camera_args = common.get_camera_args(
            data, 'pointcloud.loc', 'pointcloud.scale', device=self.device)

        # Transform GT data into camera coordinate system
        world_mat, camera_mat = camera_args['Rt'], camera_args['K']
        points_transformed = common.transform_points(points, world_mat)
        # Transform GT normals to camera coordinate system
        world_normal_mat = world_mat[:, :, :3]
        normals = common.transform_points(normals, world_normal_mat)

        outputs1, outputs2 = self.model(img, camera_mat)
        loss = self.compute_loss(
            outputs1, outputs2, points_transformed, normals, img)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def give_edges(self, pred, block_id):
        r''' Returns the edges for given block.

        Arguments:
            pred (tensor): vertex predictions of dim
                            (batch_size, n_vertices, 3)
            block_id (int): deformation block id (1,2 or 3)
        '''
        batch_size = pred.shape[0]  # (batch_size, n_vertices, 3)
        num_edges = self.edges[block_id-1].shape[0]
        edges = self.edges[block_id-1]
        nod1 = torch.index_select(pred, 1, edges[:, 0].long())
        nod2 = torch.index_select(pred, 1, edges[:, 1].long())
        assert(
            nod1.shape == (batch_size, num_edges, 3) and
            nod2.shape == (batch_size, num_edges, 3))
        final_edges = torch.sub(nod1, nod2)
        assert(final_edges.shape == (batch_size, num_edges, 3))
        return final_edges

    def edge_length_loss(self, pred, block_id):
        r''' Returns the edge length loss for given block.

        Arguments:
            pred (tensor): vertex predictions of dim
                            (batch_size, n_vertices, 3)
            block_id (int): deformation block id (1,2 or 3)
        '''
        batch_size = pred.shape[0]
        num_edges = self.edges[block_id-1].shape[0]
        final_edges = self.give_edges(pred, block_id)
        l_e = final_edges.pow(2).sum(dim=2)
        assert(l_e.shape == (batch_size, num_edges))
        l_e = l_e.mean()
        return l_e

    def give_laplacian_coordinates(self, pred, block_id):
        r''' Returns the laplacian coordinates for the predictions and given block.

            The helper matrices are used to detect neighbouring vertices and
            the number of neighbours which are relevant for the weight matrix.
            The maximal number of neighbours is 8, and if a vertex has less,
            the index -1 is used which points to the added zero vertex.

        Arguments:
            pred (tensor): vertex predictions
            block_id (int): deformation block id (1,2 or 3)
        '''
        batch_size = pred.shape[0]
        num_vert = pred.shape[1]
        # Add "zero vertex" for vertices with less than 8 neighbours
        vertex = torch.cat(
            [pred, torch.zeros(batch_size, 1, 3).to(self.device)], 1)
        assert(vertex.shape == (batch_size, num_vert+1, 3))
        # Get 8 neighbours for each vertex; if a vertex has less, the
        # remaining indices are -1
        indices = torch.from_numpy(
            self.lape_idx[block_id-1][:, :8]).to(self.device)
        assert(indices.shape == (num_vert, 8))
        weights = torch.from_numpy(
            self.lape_idx[block_id-1][:, -1]).float().to(self.device)
        weights = torch.reciprocal(weights)
        weights = weights.view(-1, 1).expand(-1, 3)
        vertex_select = vertex[:, indices.long(), :]
        assert(vertex_select.shape == (batch_size, num_vert, 8, 3))
        laplace = vertex_select.sum(dim=2)  # Add neighbours
        laplace = torch.mul(laplace, weights)  # Multiply by weights
        laplace = torch.sub(pred, laplace)  # Subtract from prediction
        assert(laplace.shape == (batch_size, num_vert, 3))
        return laplace

    def laplacian_loss(self, pred1, pred2, block_id):
        r''' Returns the Laplacian loss and move loss for given block.

        Arguments:
            pred (tensor): vertex predictions from previous block
            pred (tensor): vertex predictions from current block
            block_id (int): deformation block id (1,2 or 3)
        '''
        lap1 = self.give_laplacian_coordinates(pred1, block_id)
        lap2 = self.give_laplacian_coordinates(pred2, block_id)
        l_l = torch.sub(lap1, lap2).pow(2).sum(dim=2).mean()

        # move loss from the authors' implementation
        move_loss = 0
        if block_id != 1:
            move_loss = torch.sub(pred1, pred2).pow(2).sum(dim=2).mean()
        return l_l, move_loss

    def normal_loss(self, pred, normals, id1, block_id):
        r''' Returns the normal loss.

            First, the GT normals are selected which are the nearest
            neighbours for each predicted vertex. Next, for each edge in the
            mesh, the first node is detected and the relevant normal as well
            as the respective edge is selected. Finally, the dot product
            between these two vectors (normalsed) are calculated and the
            absolute value is taken.

        Arguments:
            pred (tensor): vertex predictions
            normals (tensor): normals of the ground truth point cloud of shape
                (batch_size, num_gt_points, 3)
            id1 (tensor): Chamfer distance IDs for predicted to GT pc of shape
                (batch_size, num_pred_pts) with values between (0, 
                num_gt_points)
            block_id (int): deformation block id (1,2 or 3)
        '''
        batch_size = pred.shape[0]
        n_verts = id1.shape[1]
        assert(pred.shape[1] == n_verts)

        help_ind = torch.arange(batch_size).view(-1, 1)
        nod1_ind = self.edges[block_id-1][:, 0]
        num_edges = nod1_ind.shape[0]
        edges = self.give_edges(pred, block_id)
        normals = normals[help_ind, id1.long(), :]
        assert(normals.size() == (batch_size, n_verts, 3))
        normals_nod1 = torch.index_select(normals, 1, nod1_ind.long())
        assert(normals_nod1.shape == (batch_size, num_edges, 3))

        normals_nod1 = F.normalize(normals_nod1, dim=2)
        edges = F.normalize(edges, dim=2)
        res = torch.mul(normals_nod1, edges).sum(dim=2).abs().mean()
        return res

    def compute_loss(self, outputs1, outputs2, gt_points, normals, img=None):
        r''' Returns the complete loss.

        The full loss is adopted from the authors' implementation and
            consists of
                a.) Chamfer distance loss
                b.) edge length loss
                c.) normal loss
                d.) Laplacian loss
                e.) move loss

        Arguments:
            outputs1 (list): first outputs of model
            outputs2 (list): second outputs of model
            gt_points (tensor): ground truth point cloud locations
            normals (tensor): normals of the ground truth point cloud
            img (tensor): input images
        '''
        pred_vertices_1, pred_vertices_2, pred_vertices_3 = outputs1

        # Chamfer Distance Loss
        lc11, lc12, id11, id12 = chamfer_distance(
            pred_vertices_1, gt_points, give_id=True)
        lc21, lc22, id21, id22 = chamfer_distance(
            pred_vertices_2, gt_points, give_id=True)
        lc31, lc32, id31, id32 = chamfer_distance(
            pred_vertices_3, gt_points, give_id=True)
        l_c = lc11.mean() + lc21.mean() + lc31.mean()
        l_c2 = lc12.mean() + lc22.mean() + lc32.mean()
        l_c = (l_c2 + self.param_chamfer_rel * l_c) * self.param_chamfer_w

        # Edge Length Loss
        l_e = (self.edge_length_loss(pred_vertices_1, 1) +
               self.edge_length_loss(pred_vertices_2, 2) +
               self.edge_length_loss(pred_vertices_3, 3)) * self.param_edge

        # Normal Loss
        l_n = (
            self.normal_loss(pred_vertices_1, normals, id11, 1) +
            self.normal_loss(pred_vertices_2, normals, id21, 2) +
            self.normal_loss(pred_vertices_3, normals, id31, 3)) * self.param_n

        # Laplacian Loss and move loss
        l_l1, _ = self.laplacian_loss(pred_vertices_1, outputs2[0], block_id=1)
        l_l2, move_loss1 = self.laplacian_loss(
            pred_vertices_2, outputs2[1], block_id=2)
        l_l3, move_loss2 = self.laplacian_loss(
            pred_vertices_3, outputs2[2], block_id=3)
        l_l = (self.param_lap_rel*l_l1 + l_l2 + l_l3) * self.param_lap
        l_m = (move_loss1 + move_loss2) * self.param_move

        # Final loss
        loss = l_c + l_e + l_n + l_l + l_m
        return loss

    def visualize(self, data):
        r''' Visualises the GT point cloud and predicted vertices (as a point cloud).

        Arguments:
            data (tensor): input data
        '''

        points_gt = data.get('pointcloud').to(self.device)
        img = data.get('inputs').to(self.device)
        camera_args = common.get_camera_args(
            data, 'pointcloud.loc', 'pointcloud.scale', device=self.device)
        world_mat, camera_mat = camera_args['Rt'], camera_args['K']

        if not os.path.isdir(self.vis_dir):
            os.mkdir(self.vis_dir)

        with torch.no_grad():
            outputs1, outputs2 = self.model(img, camera_mat)

        pred_vertices_1, pred_vertices_2, pred_vertices_3 = outputs1
        points_out = common.transform_points_back(pred_vertices_3, world_mat)
        points_out = points_out.cpu().numpy()
        input_img_path = os.path.join(self.vis_dir, 'input.png')
        save_image(img.cpu(), input_img_path, nrow=4)

        points_gt = points_gt.cpu().numpy()
        batch_size = img.size(0)
        for i in range(batch_size):
            out_file = os.path.join(self.vis_dir, '%03d.png' % i)
            out_file_gt = os.path.join(self.vis_dir, '%03d_gt.png' % i)
            vis.visualize_pointcloud(points_out[i], out_file=out_file)
            vis.visualize_pointcloud(points_gt[i], out_file=out_file_gt)

    def eval_step(self, data):
        r''' Performs an evaluation step.

        Arguments:
            data (tensor): input data
        '''
        self.model.eval()
        points = data.get('pointcloud').to(self.device)
        img = data.get('inputs').to(self.device)
        normals = data.get('pointcloud.normals').to(self.device)

        # Transform GT points to camera coordinates
        camera_args = common.get_camera_args(
            data, 'pointcloud.loc', 'pointcloud.scale', device=self.device)
        world_mat, camera_mat = camera_args['Rt'], camera_args['K']
        points_transformed = common.transform_points(points, world_mat)
        # Transform GT normals to camera coordinates
        world_normal_mat = world_mat[:, :, :3]
        normals = common.transform_points(normals, world_normal_mat)

        with torch.no_grad():
            outputs1, outputs2 = self.model(img, camera_mat)

        pred_vertices_1, pred_vertices_2, pred_vertices_3 = outputs1

        loss = self.compute_loss(
            outputs1, outputs2, points_transformed, normals, img)
        lc1, lc2, id31, id32 = chamfer_distance(
            pred_vertices_3, points_transformed, give_id=True)
        l_c = (lc1+lc2).mean()
        l_e = self.edge_length_loss(pred_vertices_3, 3)
        l_n = self.normal_loss(pred_vertices_3, normals, id31, 3)
        l_l, move_loss = self.laplacian_loss(
            pred_vertices_3, outputs2[2], block_id=3)

        eval_dict = {
            'loss': loss.item(),
            'chamfer': l_c.item(),
            'edge': l_e.item(),
            'normal': l_n.item(),
            'laplace': l_l.item(),
            'move': move_loss.item()
        }
        return eval_dict
