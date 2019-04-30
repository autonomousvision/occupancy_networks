import numpy as np
#import os
#import json
import torch
from torch.autograd import Variable
from im2mesh.dmc.utils.pointTriangleDistance import pointTriangleDistance, pointTriangleDistanceFast
from im2mesh.dmc.ops.table import get_triangle_table, get_unique_triangles, vertices_on_location
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection

eps = 1e-6
topologys = get_triangle_table()

def dis_to_meshs(pts, pts_index, vectices, x_, y_, z_ ):
    """ Return the distances from a point set to all acceptable topology types
    in a single cell
    Input:
        pts,        (Nx3) a set of points
        pts_index,  (Nx1) indicating if a point is in the cell or not
        vertices,   (3x12) the 12 vertices on each edge of the cell
        x_,         the offset of the cell in x direction
        y_,         the offset of the cell in y direction
        z_,         the offset of the cell in z direction
    Output:
        distances
    """
    distances = [dis_to_mesh(pts[0], pts_index, vectices, faces, x_, y_, z_) for faces in topologys]
    distances = torch.cat(distances)
    # adaptively assign the cost for the empty case
    if len(pts_index)!=0:
        distances[-1].item = torch.max(distances[0:-1]).item() * 10.0 
    return distances



def dis_to_mesh(pts, pts_index, vertices, faces, x_, y_, z_):
    """ Return the distance from a point set to a single topology type 
    Input:
        pts,        (Nx3) a set of points
        pts_index,  (Nx1) indicating if a point is in the cell or not
        vertices,   (3x12) the 12 vertices on each edge of the cell
        faces,      (fx3) the 
        x_,         the offset of the cell in x direction
        y_,         the offset of the cell in y direction
        z_,         the offset of the cell in z direction
    Output:
        distances
    """
    if pts.is_cuda:
        dtype = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor
    else:
        dtype = torch.FloatTensor
        dtype_long = torch.LongTensor
    one = Variable(torch.ones(1).type(dtype), requires_grad=True)
    
    if len(pts_index) == 0 and len(faces) == 0:
        return 0.0*one

    if len(pts_index) == 0 and len(faces) != 0:
        return 1.0*one

    if len(pts_index) != 0 and len(faces) == 0:
        return 1e+3 * one


    pts_index = Variable(dtype_long(pts_index))
    # for each triangles in each topology, face is a vector of 3
    dis_all_faces = []
    for face in faces:
       
        triangle = torch.cat((torch.cat(vertices[face[0]]).unsqueeze(1),
                              torch.cat(vertices[face[1]]).unsqueeze(1),
                              torch.cat(vertices[face[2]]).unsqueeze(1)), 1)
        # use the fast and approximated point to triangle distance
        dis_all_faces.append(pointTriangleDistanceFast(triangle, pts.index_select(0, pts_index) 
            - Variable(dtype([x_, y_, z_])).unsqueeze(0).expand(pts_index.size()[0], 3)))

    # only count the nearest distance to the triangles
    dis_all_faces, _ = torch.min(torch.cat(dis_all_faces, dim=1), dim=1)

    return torch.mean(dis_all_faces).view(1)



def pts_in_cell(pts, cell):
    """ get the point indices incide of a given cell (pyTorch)
    Input: 
        pts,        a set of points in pytorch format
        cell,       a list of 6 numbers {x1, y1, z1, x2, y2, z2}
    Output:
        inds,       a list of indices for points inside the cell
    """
    N = pts.size()[1] 
    cell = torch.FloatTensor(cell)
    if pts.is_cuda:
        cell = cell.cuda()
    inds = [i for i in range(N) if pts[0,i,0].item()>cell[0] and pts[0,i,0].item() < cell[3] 
                               and pts[0,i,1].item()>cell[1] and pts[0,i,1].item() < cell[4] 
                               and pts[0,i,2].item()>cell[2] and pts[0,i,2].item() < cell[5]] 
    return inds 



def pts_in_cell_numpy(pts, cell):
    """ get the point indices incide of a given cell (numpy)
    Input: 
        pts,        a set of points in numpy format
        cell,       a list of 6 numbers {x1, y1, z1, x2, y2, z2}
    Output:
        inds,       a list of indices for points inside the cell
    """
    N = pts.shape[0] 
    inds = [i for i in range(N) if pts[i,0]>cell[0] and pts[i,0] < cell[3] 
                               and pts[i,1]>cell[1] and pts[i,1] < cell[4] 
                               and pts[i,2]>cell[2] and pts[i,2] < cell[5]] 
    return inds 



def offset_to_vertices(offset, x, y, z):
    """ get 12 intersect points on each edge of a single cell """
    if offset.is_cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    one = Variable(torch.ones(1).type(dtype), requires_grad=True)
    p = [ [(0.5-offset[0, x+1, y+1, z  ])*one,    1.0*one,    0.0*one], #0
          [1.0*one,     (0.5-offset[1, x+1, y+1, z ])*one,    0.0*one], #1
          [(0.5-offset[0, x+1, y  , z  ])*one,    0.0*one,    0.0*one], #2
          [0.0*one,     (0.5-offset[1, x  , y+1, z ])*one,    0.0*one], #3

          [(0.5-offset[0, x+1, y+1, z+1])*one,    1.0*one,    1.0*one], #4
          [1.0*one,     (0.5-offset[1, x+1, y+1,z+1])*one,    1.0*one], #5
          [(0.5-offset[0, x+1, y  , z+1])*one,    0.0*one,    1.0*one], #6
          [0.0*one,     (0.5-offset[1, x  , y+1,z+1])*one,    1.0*one], #7

          [0.0*one,     1.0*one,     (0.5-offset[2, x  ,y+1,z+1])*one], #8
          [1.0*one,     1.0*one,     (0.5-offset[2, x+1,y+1,z+1])*one], #9
          [1.0*one,     0.0*one,     (0.5-offset[2, x+1,y  ,z+1])*one], #10
          [0.0*one,     0.0*one,     (0.5-offset[2, x  ,y  ,z+1])*one]] #11
    return p 


# get normal vector of all triangles dependent on the location of the cell
#    0: x1
#    1: x2
#    2: y1
#    3: y2
#    4: z1
#    5: z2
#    6: inner
def offset_to_normal(offset, x, y, z, location):
    """get normal vector of all triangles"""
    p = offset_to_vertices(offset, x, y, z)
    # get unique triangles from all topologies
    triangles, classes = get_unique_triangles(symmetry=0)

    # get normal vector of each triangle
    # assign a dummy normal vector to the unconnected ones

    # the vertices we care on the specific face
    vertices = []
    if location<6:
        vertices = vertices_on_location()[location]

    normal = []
    if offset.is_cuda:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    for tri in triangles:
        # if the triangle doesn't has a line on the face we care about 
        # simply assign a dummy normal vector
        intersection = [xx for xx in vertices if xx in tri]
        #print tri, intersection
        if location < 6 and len(intersection)!=2:
            normal.append(Variable(torch.ones(3, 1).type(dtype)))
        else:
	    ### when inside/outside is considered
            a=tri[0]
            b=tri[1]
            c=tri[2]
            normal.append(torch.cross(torch.cat(p[b])-torch.cat(p[a]), torch.cat(p[c])-torch.cat(p[a])).view(3, 1))
    return torch.cat(normal).view(-1,3)



def write_to_off(vertices, faces, filename):
    """write the given vertices and faces to off"""
    f = open(filename, 'w')
    f.write('OFF\n')

    n_vertice = vertices.shape[0] 
    n_face = faces.shape[0] 
    f.write('%d %d 0\n' % (n_vertice, n_face))
    for nv in range(n_vertice):
        ## !!! exchange the coordinates to match the orignal simplified mesh !!! 
        ## !!! need to check where the coordinate is exchanged !!!
        f.write('%f %f %f\n' % (vertices[nv,1], vertices[nv,2], vertices[nv,0]))
    for nf in range(n_face):
        f.write('3 %d %d %d\n' % (faces[nf,0], faces[nf,1], faces[nf,2]))


def gaussian_kernel(l, sig=1.):
    """ get the gaussian kernel 
    https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy 
    """
    ax = np.arange(-l//2 + 1., l//2 + 1.)
    xx, yy, zz = np.meshgrid(ax, ax, ax)

    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2. * sig**2))

    return kernel


def unique_rows(a):
    """ Return the matrix with unique rows """
    rowtype = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    b = np.ascontiguousarray(a).view(rowtype)
    _, idx, inverse = np.unique(b, return_index=True, return_inverse=True)
    return a[idx], inverse 
