cimport cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, malloc
from libcpp cimport bool
from cpython cimport PyObject, Py_INCREF

CREATE_INIT = True # workaround, so cython builds a init function

np.import_array() 


cdef extern from "offscreen.h":
  void renderDepthMesh(double *FM, int fNum, double *VM, int vNum, double *CM, double *intrinsics, int *imgSizeV, double *zNearFarV, unsigned char * imgBuffer, float *depthBuffer, bool *maskBuffer, double linewidth, bool coloring);


def render(double[:,::1] vertices, double[:,::1] faces, double[::1] cam_intr, double[::1] znf, int[::1] img_size):
  if vertices.shape[0] != 3:
    raise Exception('vertices must be a 3xM double array')
  if faces.shape[0] != 3:
    raise Exception('faces must be a 3xM double array')
  if cam_intr.shape[0] != 4:
    raise Exception('cam_intr must be a 4x1 double vector')
  if img_size.shape[0] != 2:
    raise Exception('img_size must be a 2x1 int vector')

  cdef double* VM = &(vertices[0,0])
  cdef int vNum = vertices.shape[1]
  cdef double* FM = &(faces[0,0])
  cdef int fNum = faces.shape[1]
  cdef double* intrinsics = &(cam_intr[0])
  cdef double* zNearVarV = &(znf[0])
  cdef int* imgSize = &(img_size[0])

  cdef bool coloring = False
  cdef double* CM = NULL

  depth = np.empty((img_size[1], img_size[0]), dtype=np.float32)
  mask  = np.empty((img_size[1], img_size[0]), dtype=np.uint8)
  img   = np.empty((3, img_size[1], img_size[0]), dtype=np.uint8)
  cdef float[:,::1] depth_view = depth
  cdef unsigned char[:,::1] mask_view = mask
  cdef unsigned char[:,:,::1] img_view = img
  cdef float* depthBuffer = &(depth_view[0,0])
  cdef bool* maskBuffer = <bool*> &(mask_view[0,0])
  cdef unsigned char* imgBuffer = &(img_view[0,0,0])

  renderDepthMesh(FM, fNum, VM, vNum, CM, intrinsics, imgSize, zNearVarV, imgBuffer, depthBuffer, maskBuffer, 0, coloring);

  return depth.T, mask.T, img.transpose((2,1,0))
