import os
from im2mesh.dmc import models, training, generation
from im2mesh import data


def get_model(cfg, device=None, **kwargs):
    encoder = cfg['model']['encoder']
    decoder = cfg['model']['decoder']
    c_dim = cfg['model']['c_dim']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    decoder_kwargs = cfg['model']['decoder_kwargs']

    encoder = models.encoder_dict[encoder](
        **encoder_kwargs
    )

    decoder = models.decoder_dict[decoder](
        **decoder_kwargs
    )

    model = models.DMC(decoder, encoder)
    model = model.to(device)
    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    input_type = cfg['data']['input_type']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    num_voxels = cfg['model']['num_voxels']
    weight_prior = cfg['model']['dmc_weight_prior']

    trainer = training.Trainer(
        model, optimizer, device=device, input_type=input_type,
        vis_dir=vis_dir, num_voxels=num_voxels,
        weight_prior=weight_prior,
    )
    return trainer


def get_generator(model, cfg, device, **kwargs):
    num_voxels = cfg['model']['num_voxels']

    generator = generation.Generator3D(
        model, device=device, num_voxels=num_voxels
    )
    return generator


def get_data_fields(split, cfg, **kwargs):
    with_transforms = cfg['data']['with_transforms']
    # TODO: put this into config
    pointcloud_n = 3000
    pointcloud_transform = data.SubsamplePointcloud(pointcloud_n)

    fields = {}
    fields['pointcloud'] = data.PointCloudField(
        cfg['data']['pointcloud_file'], pointcloud_transform,
        with_transforms=with_transforms
    )

    return fields
