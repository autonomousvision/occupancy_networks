import torch
import torch.nn as nn
import torch.nn.functional as F
import im2mesh.common as common
from matplotlib import pyplot as plt


class GraphUnpooling(nn.Module):
    ''' Graph Unpooling Layer.

        Unpools additional vertices following the helper file and uses the
        average feature vector from the two adjacent vertices
    '''

    def __init__(self, pool_idx_array):
        ''' Initialisation

        Args:
            pool_idx_array (tensor): vertex IDs that should be comined to new
            vertices
        '''
        super(GraphUnpooling, self).__init__()
        self.pool_x1 = pool_idx_array[:, 0]
        self.pool_x2 = pool_idx_array[:, 1]

    def forward(self, x):
        num_new_v = len(self.pool_x1)
        batch_size = x.shape[0]
        num_feats = x.shape[2]

        x1 = x[:, self.pool_x1.long(), :]
        x2 = x[:, self.pool_x2.long(), :]
        new_v = torch.add(x1, x2).mul(0.5)
        assert(new_v.shape == (batch_size, num_new_v, num_feats))
        out = torch.cat([x, new_v], dim=1)
        return out


class GraphConvolution(nn.Module):
    ''' Graph Convolution Layer

        Performs a Graph Convlution on the input vertices. The neighbouring
        vertices for each vertex are extracted by using the helper file
    '''

    def __init__(self, support_array, input_dim=963,
                 output_dim=192, bias=True, sparse=False):
        ''' Intialisation

        Args:
        support_array (tnsor): sparse weighted adjencency matrix
                with non-zero entries on the diagonal
        input_dim (int): dimension of input feature vector
        output_dim (int): dimension of output feature vector
        bias (bool): whether a bias weight should be used
        sparse (bool): if sparse matmul
        '''
        super(GraphConvolution, self).__init__()
        self.support_array = support_array.float()
        self.sparse = sparse

        self.lin = nn.Linear(input_dim, output_dim, bias=bias)
        self.lin2 = nn.Linear(input_dim, output_dim, bias=False)
        # Assume batch_size = 12
        if sparse:
            dim_full = self.support_array.size()[0]
            ind = self.support_array._indices()
            ind_ex = torch.tensor(
                [ind.cpu().numpy()+(i*dim_full) for i in range(12)]).long().to(
                    ind.get_device())
            ind_ex = ind_ex.transpose(0, 1)
            ind_ex = ind_ex.contiguous().view(2, -1)
            val = self.support_array._values()
            val_ex = val.repeat(12)
            dim_ex = torch.Size(torch.tensor(self.support_array.size())*12)
            self.exp_array = torch.sparse.FloatTensor(ind_ex, val_ex, dim_ex)
        self.dense_support = self.support_array.to_dense().float()

        # Initialise Weights
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.constant_(self.lin.bias, 0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x):
        x_1 = self.lin(x)
        batch_size = x_1.shape[0]
        num_p = x_1.shape[1]
        f_dim = x_1.shape[2]
        x_2 = self.lin2(x)
        if self.sparse and batch_size == 12:
            x_2 = x_2.view(-1, x_2.shape[2])
            res = torch.matmul(self.exp_array, x_2)
            res = res.view(batch_size, num_p, f_dim)
        else:
            res = torch.matmul(self.dense_support, x_2)
        out = torch.add(x_1, res)
        return out


class GraphProjection(nn.Module):
    """Graph Projection layer.

        Projects the predicted point cloud to the respective 2D coordinates
        given the camera and world matrix, and returns the concatenated
        features from the respective locations for each point
    """

    def __init__(self):
        super(GraphProjection, self).__init__()

    def visualise_projection(self, points_img, img, output_file='./out.png'):
        ''' Visualises the vertex projection to the image plane.

            Args:
                points_img (numpy array): points projected to the image plane
                img (numpy array): image
                output_file (string): where the result should be saved
        '''
        plt.imshow(img.transpose(1, 2, 0))
        plt.plot(
            (points_img[:, 0] + 1)*img.shape[1]/2,
            (points_img[:, 1] + 1) * img.shape[2]/2, 'x')
        plt.savefig(output_file)

    def forward(self, x, fm, camera_mat, img=None, visualise=False):
        ''' Performs a forward pass through the GP layer.

        Args:
            x (tensor): coordinates of shape (batch_size, num_vertices, 3)
            f (list): list of feature maps from where the image features
                        should be pooled
            camera_mat (tensor): camera matrices for transformation to 2D
                        image plane
            img (tensor): images (just fo visualisation purposes)
        '''
        points_img = common.project_to_camera(x, camera_mat)
        points_img = points_img.unsqueeze(1)
        feats = []
        feats.append(x)
        for fmap in fm:
            # bilinearly interpolate to get the corresponding features
            feat_pts = F.grid_sample(fmap, points_img)
            feat_pts = feat_pts.squeeze(2)
            feats.append(feat_pts.transpose(1, 2))
        # Just for visualisation purposes
        if visualise and (img is not None):
            self.visualise_projection(
                points_img.squeeze(1)[0].detach().cpu().numpy(),
                img[0].cpu().numpy())

        outputs = torch.cat([proj for proj in feats], dim=2)
        return outputs
