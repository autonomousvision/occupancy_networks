import os
from im2mesh.encoder import encoder_dict
from im2mesh.r2n2 import models, training, generation
from im2mesh import data


def get_model(cfg, device=None, **kwargs):
    ''' Return the model.

    Args:
        cfg (dict): loaded yaml config
        device (device): pytorch device
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    dim = cfg['data']['dim']
    # z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']
    # encoder_kwargs = cfg['model']['encoder_kwargs']
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

    model = models.R2N2(decoder, encoder)
    model = model.to(device)

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): R2N2 model
        optimizer (optimizer): pytorch optimizer
        cfg (dict): loaded yaml config
        device (device): pytorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']

    trainer = training.Trainer(
        model, optimizer, device=device,
        input_type=input_type, vis_dir=vis_dir,
        threshold=threshold
    )
    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): R2N2 model
        cfg (dict): loaded yaml config
        device (device): pytorch device
    '''
    generator = generation.VoxelGenerator3D(
        model, device=device
    )
    return generator


def get_data_fields(split, cfg, **kwargs):
    ''' Returns the data fields.

    Args:
        split (str): the split which should be used
        cfg (dict): loaded yaml config
    '''
    with_transforms = cfg['data']['with_transforms']

    fields = {}

    if split == 'train':
        fields['voxels'] = data.VoxelsField(
            cfg['data']['voxels_file']
        )
    elif split in ('val', 'test'):
        fields['points_iou'] = data.PointsField(
            cfg['data']['points_iou_file'],
            with_transforms=with_transforms,
            unpackbits=cfg['data']['points_unpackbits'],
        )

    return fields
