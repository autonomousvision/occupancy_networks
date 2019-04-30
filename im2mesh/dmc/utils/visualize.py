import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import os
import numpy as np
from utils.util import write_to_off, unique_rows
from _ext import eval_util


def save_mesh_fig(pts_rnd_, offset, topology, x_grids, y_grids, z_grids, ind, args, phase):
    """ save the estimated mesh with maximum likelihood as image """ 
    # get the topology type with maximum probability in each cell
    num_cells = len(x_grids)-1
    _, topology_max = torch.max(topology, dim=1)
    topology_max = topology_max.view(num_cells, num_cells, num_cells)

    # pre-allocate the memory, not safe
    vertices = torch.FloatTensor(num_cells**3 * 12, 3)
    faces = torch.FloatTensor(num_cells**3 * 12, 3)
    num_vertices = torch.LongTensor(1)
    num_faces = torch.LongTensor(1)

    # get the mesh from the estimated offest and topology
    eval_util.pred_to_mesh(offset.data.cpu(), topology_max.data.cpu(),
            vertices, faces, num_vertices, num_faces)
    if num_vertices[0] == 0:
        return

    # cut the vertices and faces matrix according to the numbers
    vertices = vertices[0:num_vertices[0], :].numpy()
    faces = faces[0:num_faces[0], :].numpy()

    # convert the vertices and face to numpy, and remove the duplicated vertices
    if len(faces):
        vertices = np.asarray(vertices)
        vertices_unique, indices = unique_rows(vertices)

        faces = np.asarray(faces).flatten()
        faces_unique = faces[indices].reshape((-1, 3))
    else:
        vertices_unique = []
        faces_unique = []


    # if save_off then skip the png figure saving for efficiency
    if phase == 'val' and args.save_off == 1:
        # exchange the axes to match the off ground truth
        if len(vertices_unique):
            vertices_unique = vertices_unique[:, [2, 0, 1]]
        write_to_off(vertices_unique, faces_unique,
                os.path.join(args.output_dir, 'mesh', '%04d.off'%ind))
    else:
        xv_cls, yv_cls, zv_cls = np.meshgrid(x_grids[:-1], y_grids[:-1], z_grids[:-1], indexing='ij')
        xv_cls = xv_cls.flatten()
        yv_cls = yv_cls.flatten()
        zv_cls = zv_cls.flatten()
        fig = plt.figure(0)
        fig.clear()
        ax = fig.add_subplot(111, projection='3d')

        # plot the scattered points
        ax.scatter(pts_rnd_[:, 0], pts_rnd_[:, 1], pts_rnd_[:, 2], '.',
            color='#727272', zorder=1)

        # plot the mesh
        color = [0.8, 0.5, 0.5]
        ax.plot_trisurf(vertices_unique[:, 0],
                        vertices_unique[:, 1],
                        vertices_unique[:, 2],
                        triangles=faces_unique,
                        color=color,
                        edgecolor='none',
                        alpha=1.0)

        ax.set_xlim(x_grids.min(), x_grids.max())
        ax.set_ylim(y_grids.min(), y_grids.max())
        ax.set_zlim(z_grids.min(), z_grids.max())
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        if phase == 'train':
            fig_name = 'train_noise%.02f_epoch%d.png' % (args.noise, ind)
        else:
            fig_name = 'val_%s_noise%.02f_ind%d.png' % (
                    os.path.splitext(os.path.basename(args.model))[0],
                    args.noise,
                    ind)
        plt.savefig(os.path.join(args.output_dir, fig_name))




def save_occupancy_fig(pts_rnd_, occupancy, x_grids, y_grids, z_grids, ind, args, phase):
    """ save the estimated occupancy as image """ 

    # skip the occupancy figure saving for efficiency
    if phase == 'val' and args.save_off == 1:
        return

    xv_cls, yv_cls, zv_cls = np.meshgrid(
            range(len(x_grids)),
            range(len(y_grids)),
            range(len(z_grids)),
            indexing='ij')
    xv_cls = xv_cls.flatten()
    yv_cls = yv_cls.flatten()
    zv_cls = zv_cls.flatten()
    fig = plt.figure(0)
    fig.clear()
    ax = fig.add_subplot(111, projection='3d')

    # plot the scattered points
    ax.scatter(pts_rnd_[:, 0], pts_rnd_[:, 1], pts_rnd_[:, 2], '.',
            color='#727272', zorder=1)

    # assign the occupancy w.r.t. the probability
    rgba_x = np.zeros((len(xv_cls), 4))
    rgba_x[:, 0] = 1.0
    rgba_x[:, 3] = occupancy.flatten()

    # plot the occupancy
    ax.scatter(xv_cls, yv_cls, zv_cls, '.', color=rgba_x, zorder=1)

    ax.set_xlim(x_grids.min(), x_grids.max())
    ax.set_ylim(y_grids.min(), y_grids.max())
    ax.set_zlim(z_grids.min(), z_grids.max())
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    if phase == 'train':
        fig = 'train_occ_noise%.02f_epoch_%d.png' % (args.noise, ind)
    else:
        fig = 'val_occ_%s_noise%.02f_ind_%d.png' % (
                os.path.splitext(os.path.basename(args.model))[0],
                args.noise,
                ind)
    plt.savefig(os.path.join(args.output_dir, fig))

