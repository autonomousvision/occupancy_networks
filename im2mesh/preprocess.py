import torch
from im2mesh import config
from im2mesh.checkpoints import CheckpointIO
from im2mesh.utils.io import export_pointcloud


class PSGNPreprocessor:
    ''' Point Set Generation Networks (PSGN) preprocessor class.

    Args:
        cfg_path (str): path to config file
        pointcloud_n (int): number of output points
        dataset (dataset): dataset
        device (device): pytorch device
        model_file (str): model file
    '''

    def __init__(self, cfg_path, pointcloud_n, dataset=None, device=None,
                 model_file=None):
        self.cfg = config.load_config(cfg_path, 'configs/default.yaml')
        self.pointcloud_n = pointcloud_n
        self.device = device
        self.dataset = dataset
        self.model = config.get_model(self.cfg, device, dataset)

        # Output directory of psgn model
        out_dir = self.cfg['training']['out_dir']
        # If model_file not specified, use the one from psgn model
        if model_file is None:
            model_file = self.cfg['test']['model_file']
        # Load model
        self.checkpoint_io = CheckpointIO(out_dir, model=self.model)
        self.checkpoint_io.load(model_file)

    def __call__(self, inputs):
        self.model.eval()
        with torch.no_grad():
            points = self.model(inputs)

        batch_size = points.size(0)
        T = points.size(1)

        # Subsample points if necessary
        if T != self.pointcloud_n:
            idx = torch.randint(
                low=0, high=T,
                size=(batch_size, self.pointcloud_n),
                device=self.device
            )
            idx = idx[:, :, None].expand(batch_size, self.pointcloud_n, 3)

            points = torch.gather(points, dim=1, index=idx)

        return points
