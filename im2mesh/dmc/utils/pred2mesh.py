import torch
import numpy as np
from im2mesh.dmc.ops.cpp_modules import pred2mesh


def unique_rows(a):
    """ Return the matrix with unique rows """
    rowtype = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    b = np.ascontiguousarray(a).view(rowtype)
    _, idx, inverse = np.unique(b, return_index=True, return_inverse=True)
    return a[idx], inverse 


def pred_to_mesh_max(offset, topology):
    """ 
    Converts the predicted offset variable and topology to a mesh by choosing the most likely  topology

    Input
    ----------
    arg1 : tensor
        offset variables [3 x W+1 x H+1 x D+1]

    arg2 : tensor
        topology probabilities [W*H*D x T]

    Returns
    -------
    trimesh format
         mesh 
    
    """ 
    # get the topology type with maximum probability in each cell
    num_cells = offset.size(1) - 1
    _, topology_max = torch.max(topology, dim=1)
    topology_max = topology_max.view(num_cells, num_cells, num_cells)

    # pre-allocate the memory, not safe
    vertices = torch.FloatTensor(num_cells**3 * 12, 3)
    faces = torch.FloatTensor(num_cells**3 * 12, 3)
    num_vertices = torch.LongTensor(1)
    num_faces = torch.LongTensor(1)
    topology_max = topology_max.int()

    # get the mesh from the estimated offest and topology
    pred2mesh.pred2mesh(offset.cpu(), topology_max.cpu(),
            vertices, faces, num_vertices, num_faces)
    
    # cut the vertices and faces matrix according to the numbers
    vertices = vertices[0:num_vertices[0], :].numpy()
    faces = faces[0:num_faces[0], :].numpy()

    # convert the vertices and face to numpy, and remove the duplicated vertices
    vertices_unique = np.asarray(vertices)
    faces_unique = np.asarray(faces)
    # if len(faces):
    #     vertices = np.asarray(vertices)
    #     vertices_unique, indices = unique_rows(vertices)

    #     faces = np.asarray(faces).flatten()
    #     faces_unique = faces[indices].reshape((-1, 3))
    # else:
    #     vertices_unique = []
    #     faces_unique = []


    # if len(vertices_unique):
    #     vertices_unique = vertices_unique[:, [2, 0, 1]]

    return vertices_unique, faces_unique
