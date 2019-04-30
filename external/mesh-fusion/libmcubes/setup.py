# -*- encoding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from Cython.Build import cythonize

import numpy
from distutils.extension import Extension

# Get the version number.
numpy_include_dir = numpy.get_include()

mcubes_module = Extension(
    "mcubes",
    [
        "mcubes.pyx",
        "pywrapper.cpp",
        "marchingcubes.cpp"
    ],
    language="c++",
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

setup(name="PyMCubes",
    version="0.0.6",
    description="Marching cubes for Python",
    author="Pablo Márquez Neila",
    author_email="pablo.marquezneila@epfl.ch",
    url="https://github.com/pmneila/PyMCubes",
    license="BSD 3-clause",
    long_description="""
    Marching cubes for Python
    """,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    packages=["mcubes"],
    ext_modules=cythonize(mcubes_module),
    requires=['numpy', 'Cython', 'PyCollada'],
    setup_requires=['numpy', 'Cython']
)
