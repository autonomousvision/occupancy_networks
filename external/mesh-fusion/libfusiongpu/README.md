# PyFusion

PyFusion is a Python framework for volumetric depth fusion.
It contains simple occupancy and TSDF fusion methods that can be executed on a CPU as well as on a GPU.

To use the code, first compile the native code via 

```bash
cd build
cmake ..
make
```
Afterwards you can compile the Cython code via

```bash
python setup.py build_ext --inplace
```

You can then use the fusion functions

```python
import pyfusion

# create a views object
# depthmaps: a NxHxW numpy float tensor of N depthmaps, invalid depth values are marked by negative numbers
# Ks: the camera intric matrices, Nx3x3 float tensor
# Rs: the camera rotation matrices, Nx3x3 float tensor
# Ts: the camera translation vectors, Nx3 float tensor
views = pyfusion.PyViews(depthmaps, Ks,Rs,Ts)

# afterwards you can fuse the depth maps for example by
# depth,height,width: number of voxels in each dimension
# truncation: TSDF truncation value
tsdf = pyfusion.tsdf_gpu(views, depth,height,width, vx_size, truncation, False)

# the same code can also be run on the CPU
tsdf = pyfusion.tsdf_cpu(views, depth,height,width, vx_size, truncation, False, n_threads=8)
```

Make sure `pyfusion` is in your `$PYTHONPATH`.
