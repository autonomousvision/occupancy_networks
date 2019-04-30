import torch
import numpy as np
from im2mesh.utils.voxels import VoxelGrid


class VoxelGenerator3D(object):
    ''' Generator class for R2N2 model.

    The output of the model is transformed to a voxel grid and returned as a
    mesh.

    Args:
        model (nn.Module): (trained) R2N2 model
        threshold (float): threshold value for deciding whether a voxel is
            occupied or not
        device (device): pytorch device
    '''

    def __init__(self, model, threshold=0.5, device=None):
        self.model = model.to(device)
        self.threshold = threshold
        self.device = device

    def generate_mesh(self, data):
        ''' Generates the output mesh.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        device = self.device

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)

        with torch.no_grad():
            out = self.model(inputs).squeeze(1).squeeze(0)

        out = out.cpu().numpy()
        mesh = self.extract_mesh(out)

        return mesh

    def extract_mesh(self, values):
        ''' Extracts the mesh.

        Args:
            values (numpy array): predicted values
        '''
        # Convert threshold to logits
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        # Extract mesh
        mesh = VoxelGrid(values >= threshold).to_mesh()

        return mesh
