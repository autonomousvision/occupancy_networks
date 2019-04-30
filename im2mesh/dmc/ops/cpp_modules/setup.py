from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='pred2mesh',
    ext_modules=[
        CppExtension('pred2mesh', [
            'pred_to_mesh_.cpp',
           # 'commons.cpp'
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })