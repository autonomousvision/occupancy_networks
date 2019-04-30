import torch
import numpy as np
import trimesh
from im2mesh.dmc.utils.pred2mesh import pred_to_mesh_max 
from im2mesh.dmc.ops.occupancy_to_topology import OccupancyToTopology
from im2mesh.dmc.ops.table import get_accept_topology


class Generator3D(object):
    def __init__(self, model, device=None, num_voxels=32):
        self.model = model.to(device)
        self.device = device
        self.num_voxels = num_voxels
        self.vis_topology = torch.LongTensor(get_accept_topology(4))

    def generate_mesh(self, data):
        self.model.eval()
        device = self.device

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)

        inputs = self.num_voxels * (inputs / 1.2 + 0.5)

        with torch.no_grad():
            offset, topology, occupancy = self.model(inputs)

        offset = offset.squeeze()
        topology = topology.squeeze()
        topology = topology[:, self.vis_topology]

        vertices, faces = pred_to_mesh_max(offset, topology)
        faces = faces.astype(np.int64)

        vertices = 1.2 * (vertices / self.num_voxels - 0.5)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        return mesh


