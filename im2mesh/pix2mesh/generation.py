import torch
import trimesh
import im2mesh.common as common


class Generator3D(object):
    ''' Mesh Generator Class for the Pixel2Mesh model.

    A forward pass is made with the image and camera matrices to obtain the
    predicted vertex locations for the mesh. Subsequently, the faces of the
    base mesh of an ellipsoid are used together with the predicted vertices to
    obtain the final mesh
    '''

    def __init__(self, model, base_mesh, device=None):
        ''' Initialisation

        Args:
            model (PyTorch model): the Pixel2Mesh model
            base_mesh (tensor): the base ellipsoid provided by the authors
            device (PyTorch device): the PyTorch device
        '''
        self.model = model.to(device)
        self.device = device
        self.base_mesh = base_mesh

    def generate_mesh(self, data, fix_normals=False):
        ''' Generates a mesh.

        Arguments:
            data (tensor): input data
            fix_normals (boolean): if normals should be fixed
        '''

        img = data.get('inputs').to(self.device)
        camera_args = common.get_camera_args(
            data, 'pointcloud.loc', 'pointcloud.scale', device=self.device)
        world_mat, camera_mat = camera_args['Rt'], camera_args['K']
        with torch.no_grad():
            outputs1, outputs2 = self.model(img, camera_mat)
            out_1, out_2, out_3 = outputs1

        transformed_pred = common.transform_points_back(out_3, world_mat)
        vertices = transformed_pred.squeeze().cpu().numpy()

        faces = self.base_mesh[:, 1:]  # remove the f's in the first column
        faces = faces.astype(int) - 1  # To adjust indices to trimesh notation
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if fix_normals:
            # Fix normals due to wrong base ellipsoid
            trimesh.repair.fix_normals(mesh)
        return mesh

    def generate_pointcloud(self, data):
        ''' Generates a pointcloud by only returning the vertices

        Arguments:
            data (tensor): input data
        '''

        img = data.get('inputs').to(self.device)
        camera_args = common.get_camera_args(
            data, 'pointcloud.loc', 'pointcloud.scale', device=self.device)
        world_mat, camera_mat = camera_args['Rt'], camera_args['K']

        with torch.no_grad():
            outputs1, _ = self.model(img, camera_mat)
            _, _, out_3 = outputs1
        transformed_pred = common.transform_points_back(out_3, world_mat)
        pc_out = transformed_pred.squeeze().cpu().numpy()
        return pc_out
