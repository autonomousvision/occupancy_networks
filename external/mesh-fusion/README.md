# Watertight and Simplified Meshes through TSDF Fusion

This repository contains a simply Python pipeline for obtaining watertight
and simplified meshes from arbitrary triangular meshes, given in `.off` format.
The approach is largely based on adapted versions of Gernot Riegler's
[pyrender](https://github.com/griegler/pyrender) and [pyfusion](https://github.com/griegler/pyfusion);
it also uses [PyMCubes](https://github.com/pmneila/PyMCubes).

If you use any of this code, please make sure to cite the following work:

    @article{Stutz2018ARXIV,
        author    = {David Stutz and Andreas Geiger},
        title     = {Learning 3D Shape Completion under Weak Supervision},
        journal   = {CoRR},
        volume    = {abs/1805.07290},
        year      = {2018},
        url       = {http://arxiv.org/abs/1805.07290},
    }

See the GitHub repositories above for additional citations.
Also check the corresponding [project page](http://davidstutz.de/projects/shape-completion/).

The pipeline consists of three steps:

1. Scaling using `1_scale.py`, which scales the meshes into `[-0.5,0.5]^3` with optional padding.
2. Rendering and fusion using `2_fusion.py` which in a first step renders views of each mesh, and uses these to peform TSDF fusion.
3. Simplification using MeshLab.

The result is illustrated below; note that the shape gets slightly thicker and
the axes change (due to marching cubes).

![Example of watertight and simplified mesh.](screenshot.jpg?raw=true "Example of watertight and simplified mesh.")

## Installation

The pipeline is mostly self-contained, given a working installation of Python.
Additionally, it requires [MeshLab](http://www.meshlab.net/) for simplification.

The used [pyfusion](https://github.com/griegler/pyfusion) library requires
CMake and Cython; optionally it uses CUDA or OpenMP for efficiency.
[pyrender](https://github.com/griegler/pyrender) requires Cython, as well.
Additionally, it requires OpenGL, GLUT and GLEW, see `librender/setup.py`
for details. [PyMCubes](https://github.com/pmneila/PyMCubes) requires Cython.
All three libraries are included in this repository.

For building follow (illustrated for the GPU version):

    # build pyfusion
    # use libfusioncpu alternatively!
    cd libfusiongpu
    mkdir build
    cd build
    cmake ..
    make
    cd ..
    python setup.py build_ext --inplace
    
    cd ..
    # build pyrender
    cd librender
    python setup.py build_ext --inplace
    
    cd ..
    # build PyMCubes
    cd libmcubes
    python setup.py build_ext --inplace

## Usage

Usage is illustrated on the shipped examples in `examples/0_in` taken 
from [ModelNet](http://modelnet.cs.princeton.edu/).

First, scale the models using:

    python 1_scale.py --in_dir=examples/0_in/ --out_dir=examples/1_scaled/

Now the models can be rendered, per default, 100 views (uniformly sampled
on a sphere) will be used:

    2_fusion.py --mode=render --in_dir=examples/1_scaled/ --depth_dir=examples/2_depth/ --out_dir=examples/2_watertight/

The details of rendering can be controlled using the following options:

    --n_views N_VIEWS     Number of views per model.
    --image_height IMAGE_HEIGHT
                          Depth image height.
    --image_width IMAGE_WIDTH
                          Depth image width.
    --focal_length_x FOCAL_LENGTH_X
                          Focal length in x direction.
    --focal_length_y FOCAL_LENGTH_Y
                          Focal length in y direction.
    --principal_point_x PRINCIPAL_POINT_X
                          Principal point location in x direction.
    --principal_point_y PRINCIPAL_POINT_Y
                          Principal point location in y direction.
    --depth_offset_factor DEPTH_OFFSET_FACTOR
                          The depth maps are offsetted using
                          depth_offset_factor*voxel_size.
    --resolution RESOLUTION
                          Resolution for fusion.
    --truncation_factor TRUNCATION_FACTOR
                          Truncation for fusion is derived as
                          truncation_factor*voxel_size.

During rendering, a small offset is added to the depth maps. This is particular
important for meshes with thin details, as for example the provided chairs.
Essentially, this thickens the structures. In the code, the offset is computed as

    voxel_size = 1/resolution
    offset = depth_offset_factor*voxel_size

Now, fusion can be run using

    python 2_fusion.py --mode=fuse --in_dir=examples/1_scaled/ --depth_dir=examples/2_depth/ --out_dir=examples/2_watertight/

For fusion, the resolution and the truncation factor are most importance.
In practice, the truncation factor may be in the range of `[0, ..., 15]`;
then, the truncation threshold is computed as

    voxel_size = 1/resolution
    truncation = truncatioN_factor*voxel_size

**Note that rendering and fusion is splitted as rendering might not work on all machines,
especially remotely (e.g. through ssh) on machines without monitor.**

Finally, simplification is performed using `meshlabserver`; make sure to have
it installed and run

    python 3_1_simplify.py --in_dir=examples/2_watertight/ --out_dir=examples/3_out/

The result of all steps is illustrated in the screenshot above.

## License

License for source code corresponding to:

D. Stutz, A. Geiger. **Learning 3D Shape Completion under Weak Supervision.** International Journal of Computer Vision (2018).

Note that the source code and/or data is based on the following projects for which separate licenses apply:

* [pyrender](https://github.com/griegler/pyrender)
* [pyfusion](https://github.com/griegler/pyfusion)
* [PyMCubes](https://github.com/pmneila/PyMCubes)

Copyright (c) 2018 David Stutz, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the corresponding papers (see above) in documents and papers that report on research using the Software.
