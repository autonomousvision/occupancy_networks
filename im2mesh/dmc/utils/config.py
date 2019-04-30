import argparse
import os
from ConfigParser import SafeConfigParser
from im2mesh.dmc.ops.table import get_triangle_table

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    # arguments for the model
    parser.add_argument('--encoder_type',
		        help='Specify the type of the encoder, e.g. point, image',
                        default='point', choices=['point', 'voxel'], type=str)
    parser.add_argument('--model',
                        help='Snapshotted model, for resuming training or validation',
                        default='', type=str)
    parser.add_argument('--num_voxels',
                        help='Number of voxels on each dimension',
                        default=8, type=int)

    # arguments for the training/test data
    parser.add_argument('--data_type',
                        help='Specify the type of the input points, e.g. ellipsoid, cube, mix',
                        default='ellipsoid', choices=['ellipsoid', 'cube', 'mix3d', 'shapenet'], type=str)
    parser.add_argument('--num_sample',
                        help='Number of sampled points on each mesh',
                        default=3000, type=int)
    parser.add_argument('--bias_data',
                        help='Boolean variable to decide if use data sampled with angle bias',
                        default=False, type=str2bool)
    parser.add_argument('--bias_level',
                        help='Float variable to decide the level of the angle bias, 1 means no data in a specific region',
                        default=0.0, type=float)
    parser.add_argument('--noise',
                        help='Scale of the noise added to the input data',
                        default=0.01, type=float)
    parser.add_argument('--perturb',
                        help='Scale of the noise added to the input data',
                        default=0, type=int)
    parser.add_argument('--noise_gt',
                        help='If true then train the network with noisy gt',
                        default=0, type=int)

    # arguments for directories and cached file names
    parser.add_argument('--data_dir',
                        help='Input directory',
                        default='./data_blob', type=str)
    parser.add_argument('--output_dir',
                        help='Output directory',
                        default=None, type=str)
    parser.add_argument('--cached_train',
                        help='Location of training data, full path is needed',
                        default=None, type=str)
    parser.add_argument('--cached_val',
                        help='Location of validation data, full path is needed',
                        default=None, type=str)

    # arguments for training process
    parser.add_argument('--batchsize',
                        help='Batch size for training',
                        default=8, type=int)
    parser.add_argument('--epoch',
                        help='Number of epochs',
                        default=510, type=int)
    parser.add_argument('--snapshot',
                        help='Snapshotting intervals',
                        default=20, type=int)
    parser.add_argument('--verbose',
                        help='If print the training details or not',
                        default=True, type=str2bool)
    parser.add_argument('--verbose_interval',
                        help='Interval of printing the training information',
                        default=10, type=int)

    # arguments for saving results
    parser.add_argument('--save_model',
                        help='If save the model or not',
                        default=True, type=str2bool)
    parser.add_argument('--save_off',
                        help='If save the off mesh during the validation phase or not',
                        default=0, type=int)

    # parameters for grid searching
    parser.add_argument('--weight_distance',
                        default=5.0, type=float)
    parser.add_argument('--weight_prior_pos',
                        default=0.2, type=float)
    parser.add_argument('--weight_prior',
                        default=10.0, type=float)
    parser.add_argument('--weight_smoothness',
                        default=3.0, type=float)
    parser.add_argument('--weight_curvature',
                        default=3.0, type=float)
    parser.add_argument('--weight_decay',
                        default=1e-3, type=float)
    parser.add_argument('--learning_rate',
                        default=1e-4, type=float)

    args = parser.parse_args()

    # note the voxel and the cells are dual representation
    # the corners of the cells are the middle point of the
    # voxels
    args.num_cells = args.num_voxels - 1
    resolution = '%dx%dx%d' % (args.num_voxels, args.num_voxels, args.num_voxels)

    if args.cached_train is None:
        args.cached_train = os.path.join(
                args.data_dir, 'points_%s_%s_train.npy' % (
                args.data_type, resolution))
    if args.cached_val is None:
        args.cached_val = os.path.join(
                args.data_dir, 'points_%s_%s_val.npy' % (
                args.data_type, resolution))

    default_dir = './output'
    if args.output_dir is None:
        args.output_dir = os.path.join(default_dir, '%s_%s_%s_pts%d_noisygt%d'
                % (args.encoder_type, args.data_type, args.num_cells,
                args.num_sample, args.noise_gt))

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.save_off and not os.path.isdir(os.path.join(args.output_dir, 'mesh')):
        os.makedirs(os.path.join(args.output_dir, 'mesh'))

    ## only support voxel input for shapenet
    if args.encoder_type == 'voxel':
        assert args.data_type == 'shapenet', 'Invalid combination of encoder_type and data_type!'

    # get the number of accepted topologies
    args.num_topology = len(get_triangle_table())

    # fix the length of the cell edge as 1.0
    args.len_cell = 1.0

    # use skip connection in the u-net
    args.skip_connection = True

    args.num_train = 200 
    args.num_val = 100

    return args
