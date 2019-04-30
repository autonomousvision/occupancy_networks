import os
from im2mesh.encoder import encoder_dict
from im2mesh.psgn import models, training, generation
from im2mesh import data


def get_model(cfg, device=None, **kwargs):
    r''' Returns the model instance.

    Args:
        cfg (yaml object): the config file
        device (PyTorch device): the PyTorch device
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']

    decoder = models.decoder_dict[decoder](
        dim=dim, c_dim=c_dim,
        **decoder_kwargs
    )

    encoder = encoder_dict[encoder](
        c_dim=c_dim,
        **encoder_kwargs
    )

    model = models.PCGN(decoder, encoder)
    model = model.to(device)
    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    r''' Returns the trainer instance.

    Args:
        model (nn.Module): PSGN model
        optimizer (PyTorch optimizer): The optimizer that should be used
        cfg (yaml object): the config file
        device (PyTorch device): the PyTorch device
    '''
    input_type = cfg['data']['input_type']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')

    trainer = training.Trainer(
        model, optimizer, device=device, input_type=input_type,
        vis_dir=vis_dir
    )
    return trainer


def get_generator(model, cfg, device, **kwargs):
    r''' Returns the generator instance.

    Args:
        cfg (yaml object): the config file
        device (PyTorch device): the PyTorch device
    '''
    generator = generation.Generator3D(model, device=device)
    return generator


def get_data_fields(mode, cfg, **kwargs):
    r''' Returns the data fields.

    Args:
        mode (string): The split that is used (train/val/test)
        cfg (yaml object): the config file
    '''
    with_transforms = cfg['data']['with_transforms']
    pointcloud_transform = data.SubsamplePointcloud(
        cfg['data']['pointcloud_target_n'])

    fields = {}
    fields['pointcloud'] = data.PointCloudField(
        cfg['data']['pointcloud_file'], pointcloud_transform,
        with_transforms=with_transforms
    )

    if mode in ('val', 'test'):
        pointcloud_chamfer_file = cfg['data']['pointcloud_chamfer_file']
        if pointcloud_chamfer_file is not None:
            fields['pointcloud_chamfer'] = data.PointCloudField(
                pointcloud_chamfer_file
            )

    return fields
