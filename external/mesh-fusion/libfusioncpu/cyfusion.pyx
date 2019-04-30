cimport cython
import numpy as np
cimport numpy as np

from libc.stdlib cimport free, malloc
from libcpp cimport bool
from cpython cimport PyObject, Py_INCREF

CREATE_INIT = True # workaround, so cython builds a init function

np.import_array()



cdef extern from "fusion.h":
  cdef cppclass Views:
    Views()
    int n_views_;
    float* depthmaps_;
    int rows_;
    int cols_;
    float* Ks_;
    float* Rs_;
    float* Ts_;

  cdef cppclass Volume:
    Volume()
    int channels_;
    int depth_;
    int height_;
    int width_;
    float* data_;


  void fusion_projectionmask_cpu(const Views& views, float vx_size, bool unknown_is_free, int n_threads, Volume& vol);
  void fusion_occupancy_cpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, int n_threads, Volume& vol);
  void fusion_tsdfmask_cpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, int n_threads, Volume& vol);
  void fusion_tsdf_cpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, int n_threads, Volume& vol);
  void fusion_tsdf_hist_cpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, float* bin_centers, int n_bins, bool unobserved_is_occupied, int n_threads, Volume& vol);

cdef class PyViews:
  cdef Views views
  # need to keep reference, otherwise it could get garbage collected
  cdef float[:,:,::1] depthmaps_ 
  cdef float[:,:,::1] Ks_
  cdef float[:,:,::1] Rs_
  cdef float[:,::1] Ts_

  def __init__(self, float[:,:,::1] depthmaps, float[:,:,::1] Ks, float[:,:,::1] Rs, float[:,::1] Ts):
    cdef int n = depthmaps.shape[0]
    if n != Ks.shape[0]:
      raise Exception('number of depthmaps and Ks differ')
    if n != Rs.shape[0]:
      raise Exception('number of depthmaps and Rs differ')
    if n != Ts.shape[0]:
      raise Exception('number of depthmaps and Ts differ')

    if Ks.shape[1] != 3 or Ks.shape[2] != 3:
      raise Exception('Ks have to be nx3x3')
    if Rs.shape[1] != 3 or Rs.shape[2] != 3:
      raise Exception('Rs have to be nx3x3')
    if Ts.shape[1] != 3:
      raise Exception('Ts have to be nx3')

    self.depthmaps_ = depthmaps
    self.Ks_ = Ks
    self.Rs_ = Rs
    self.Ts_ = Ts
    
    self.views.depthmaps_ = &(depthmaps[0,0,0])  
    self.views.n_views_ = depthmaps.shape[0]
    self.views.rows_ = depthmaps.shape[1]
    self.views.cols_ = depthmaps.shape[2]
    self.views.Ks_ = &(Ks[0,0,0])  
    self.views.Rs_ = &(Rs[0,0,0]) 
    self.views.Ts_ = &(Ts[0,0]) 


cdef class PyVolume:
  cdef Volume vol

  def __init__(self, float[:,:,:,::1] data):
    self.vol = Volume()
    self.vol.data_ = &(data[0,0,0,0])
    self.vol.channels_ = data.shape[0]
    self.vol.depth_ = data.shape[1]
    self.vol.height_ = data.shape[2]
    self.vol.width_ = data.shape[3]


def projmask_cpu(PyViews views, int depth, int height, int width, float vx_size, bool unknown_is_free, int n_threads=8):
  vol = np.empty((1, depth, height, width), dtype=np.float32)
  cdef float[:,:,:,::1] vol_view = vol
  cdef PyVolume py_vol = PyVolume(vol_view)
  fusion_projectionmask_cpu(views.views, vx_size, unknown_is_free, n_threads, py_vol.vol)
  return vol

def occupancy_cpu(PyViews views, int depth, int height, int width, float vx_size, float truncation, bool unknown_is_free, int n_threads=8):
  vol = np.empty((1, depth, height, width), dtype=np.float32)
  cdef float[:,:,:,::1] vol_view = vol
  cdef PyVolume py_vol = PyVolume(vol_view)
  fusion_occupancy_cpu(views.views, vx_size, truncation, unknown_is_free, n_threads, py_vol.vol)
  return vol

def tsdfmask_cpu(PyViews views, int depth, int height, int width, float vx_size, float truncation, bool unknown_is_free, int n_threads=8):
  vol = np.empty((1, depth, height, width), dtype=np.float32)
  cdef float[:,:,:,::1] vol_view = vol
  cdef PyVolume py_vol = PyVolume(vol_view)
  fusion_tsdfmask_cpu(views.views, vx_size, truncation, unknown_is_free, n_threads, py_vol.vol)
  return vol

def tsdf_cpu(PyViews views, int depth, int height, int width, float vx_size, float truncation, bool unknown_is_free, int n_threads=8):
  vol = np.empty((1, depth, height, width), dtype=np.float32)
  cdef float[:,:,:,::1] vol_view = vol
  cdef PyVolume py_vol = PyVolume(vol_view)
  fusion_tsdf_cpu(views.views, vx_size, truncation, unknown_is_free, n_threads, py_vol.vol)
  return vol

def tsdf_hist_cpu(PyViews views, int depth, int height, int width, float vx_size, float truncation, bool unknown_is_free, float[::1] bins, bool unobserved_is_occupied=True, int n_threads=8):
  cdef int n_bins = bins.shape[0]
  vol = np.empty((n_bins, depth, height, width), dtype=np.float32)
  cdef float[:,:,:,::1] vol_view = vol
  cdef PyVolume py_vol = PyVolume(vol_view)
  fusion_tsdf_hist_cpu(views.views, vx_size, truncation, unknown_is_free, &(bins[0]), n_bins, unobserved_is_occupied, n_threads, py_vol.vol)
  return vol