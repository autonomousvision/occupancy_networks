from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='_cuda_ext',
    ext_modules=[
        CUDAExtension('_cuda_ext', [
            'src/extension.cpp',
            'src/curvature_constraint_kernel.cu',
            'src/grid_pooling_kernel.cu',
            'src/occupancy_to_topology_kernel.cu',
            'src/occupancy_connectivity_kernel.cu',
            'src/point_triangle_distance_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
