import os
from im2mesh.encoder import encoder_dict
from im2mesh.pix2mesh import models, training, generation
from im2mesh import data
import pickle
import numpy as np


def get_model(cfg, device=None, **kwargs):
    ''' Returns the Pixel2Mesh model.

    Args:
        cfg (yaml file): config file
        device (PyTorch device): PyTorch device
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    feat_dim = cfg['model']['feat_dim']
    hidden_dim = cfg['model']['hidden_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    # Encoding necessary due to python2 pickle to python3 pickle convert
    ellipsoid = pickle.load(
        open(cfg['data']['ellipsoid'], 'rb'), encoding='latin1')

    decoder = models.decoder_dict[decoder](
        ellipsoid, device=device, hidden_dim=hidden_dim, feat_dim=feat_dim,
        **decoder_kwargs
    )

    encoder = encoder_dict[encoder](
        return_feature_maps=True,
        **encoder_kwargs
    )

    model = models.Pix2Mesh(decoder, encoder)
    model = model.to(device)
    return model


def get_trainer(model, optimizer, cfg, device):
    ''' Return the trainer object for the Pixel2Mesh model.
    Args:
        model (PyTorch model): Pixel2Mesh model
        optimizer( PyTorch optimizer): The optimizer that should be used
        cfg (yaml file): config file
        device (PyTorch device): The PyTorch device that should be used
    '''
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    adjust_losses = cfg['model']['adjust_losses']
    # Encoding necessary due to python2 pickle to python3 pickle convert
    ellipsoid = pickle.load(
        open(cfg['data']['ellipsoid'], 'rb'), encoding='latin1')
    trainer = training.Trainer(
        model, optimizer, ellipsoid, vis_dir, device=device, adjust_losses=adjust_losses)
    return trainer


def get_generator(model, cfg, device):
    ''' Returns a generator object for the Pixel2Mesh model.

    Args:
        model (PyTorch model): Pixel2Mesh model
        cfg (yaml file): config file
        device (PyTorch device): The PyTorch device that should be used
    '''
    base_mesh = np.loadtxt(cfg['data']['base_mesh'], dtype='|S32')
    generator = generation.Generator3D(
        model, base_mesh, device=device)
    return generator


def get_data_fields(mode, cfg):
    ''' Returns the respective data fields.

    Args:
        mode (string): which split should be performed (train/test)
        cfg (yaml file): config file
    '''
    with_transforms = cfg['data']['with_transforms']
    pointcloud_transform = data.SubsamplePointcloud(
        cfg['data']['pointcloud_target_n'])
    fields = {}
    fields['pointcloud'] = data.PointCloudField(
        cfg['data']['pointcloud_file'], pointcloud_transform,
        with_transforms=with_transforms)

    return fields
