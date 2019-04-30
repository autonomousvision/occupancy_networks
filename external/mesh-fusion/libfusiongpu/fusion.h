#ifndef FUSION_H
#define FUSION_H

#include <cstdio>
#include <cmath>

#ifdef __CUDA_ARCH__
#define FUSION_FUNCTION __host__ __device__
#else
#define FUSION_FUNCTION
#endif


class Views {
public:
  int n_views_;
  float* depthmaps_;
  int rows_;
  int cols_;
  float* Ks_;
  float* Rs_;
  float* Ts_;

  Views() : n_views_(0), depthmaps_(0), rows_(0), cols_(0), Ks_(0), Rs_(0), Ts_(0) {}
};

class Volume {
public:
  int channels_;
  int depth_;
  int height_;
  int width_;
  float* data_;
  
  Volume() : channels_(0), depth_(0), height_(0), width_(0), data_(0) {}
};

FUSION_FUNCTION
inline int volume_idx(const Volume* vol, int c, int d, int h, int w) {
  return ((c * vol->depth_ + d) * vol->height_ + h) * vol->width_ + w;
}

FUSION_FUNCTION
inline float volume_get(const Volume* vol, int c, int d, int h, int w) {
  return vol->data_[volume_idx(vol, c,d,h,w)];
}

FUSION_FUNCTION
inline void volume_set(const Volume* vol, int c, int d, int h, int w, float val) {
  vol->data_[volume_idx(vol, c,d,h,w)] = val;
}

FUSION_FUNCTION
inline void volume_add(const Volume* vol, int c, int d, int h, int w, float val) {
  vol->data_[volume_idx(vol, c,d,h,w)] += val;
}

FUSION_FUNCTION
inline void volume_div(const Volume* vol, int c, int d, int h, int w, float val) {
  vol->data_[volume_idx(vol, c,d,h,w)] /= val;
}

struct FusionFunctor {
  FUSION_FUNCTION
  virtual bool new_sample(Volume* vol, float vx_depth, float dm_depth, int d, int h, int w, int* n_valid_views) const {
    return false;
  }
  
  FUSION_FUNCTION
  virtual void before_sample(Volume* vol, int d, int h, int w) const {
    for(int c = 0; c < vol->channels_; ++c) {
      volume_set(vol, c,d,h,w, 0);
    }
  }

  FUSION_FUNCTION
  virtual void after_sample(Volume* vol, int d, int h, int w, int n_valid_views) const {}
};




FUSION_FUNCTION
inline void fusion_idx2dhw(int idx, int width, int height, int& d, int& h, int &w) {
  w = idx % (width);
  d = idx / (width * height);
  h = ((idx - w) / width) % height;
} 

FUSION_FUNCTION
inline void fusion_dhw2xyz(int d, int h, int w, float vx_size, float& x, float& y, float& z) {
  // +0.5: move vx_center from (0,0,0) to (0.5,0.5,0.5), therefore vx range in [0, 1)
  // *vx_size: scale from [0,vx_resolution) to [0,1)
  // -0.5: move box to center, resolution [-.5,0.5)
  x = ((w + 0.5) * vx_size) - 0.5;
  y = ((h + 0.5) * vx_size) - 0.5;
  z = ((d + 0.5) * vx_size) - 0.5;
}

FUSION_FUNCTION
inline void fusion_project(const Views* views, int vidx, float x, float y, float z, float& u, float& v, float& d) {
  float* K = views->Ks_ + vidx * 9;
  float* R = views->Rs_ + vidx * 9;
  float* T = views->Ts_ + vidx * 3;

  float xt = R[0] * x + R[1] * y + R[2] * z + T[0];
  float yt = R[3] * x + R[4] * y + R[5] * z + T[1];
  float zt = R[6] * x + R[7] * y + R[8] * z + T[2];
  // printf("  vx has center %f,%f,%f and projects to %f,%f,%f\n", x,y,z, xt,yt,zt);

  u = K[0] * xt + K[1] * yt + K[2] * zt;
  v = K[3] * xt + K[4] * yt + K[5] * zt;
  d = K[6] * xt + K[7] * yt + K[8] * zt;
  u /= d;
  v /= d;
}


struct ProjectionMaskFusionFunctor : public FusionFunctor {
  bool unknown_is_free_;
  ProjectionMaskFusionFunctor(bool unknown_is_free) :
    unknown_is_free_(unknown_is_free) {}

  FUSION_FUNCTION
  virtual bool new_sample(Volume* vol, float vx_depth, float dm_depth, int d, int h, int w, int* n_valid_views) const {
    if(unknown_is_free_ && dm_depth < 0) {
      dm_depth = 1e9;
    }
    if(dm_depth > 0) {
      volume_set(vol, 0,d,h,w, 1);
      return false;
    }
    return true;
  }
};

void fusion_projectionmask_gpu(const Views& views, float vx_size, bool unknown_is_free, Volume& vol);


struct OccupancyFusionFunctor : public FusionFunctor {
  float truncation_;
  bool unknown_is_free_;
  OccupancyFusionFunctor(float truncation, bool unknown_is_free) :
    truncation_(truncation), unknown_is_free_(unknown_is_free) {}

  FUSION_FUNCTION
  virtual void before_sample(Volume* vol, int d, int h, int w) const {
    for(int c = 0; c < vol->channels_; ++c) {
      volume_set(vol, c,d,h,w, 1);
    }
  }

  FUSION_FUNCTION
  virtual bool new_sample(Volume* vol, float vx_depth, float dm_depth, int d, int h, int w, int* n_valid_views) const {
    if(unknown_is_free_ && dm_depth < 0) {
      dm_depth = 1e9;
    }
    float diff = dm_depth - vx_depth;
    if(dm_depth > 0 && diff > truncation_) {
      volume_set(vol, 0,d,h,w, 0);
      return false;
    }
    return true;
  }
};

void fusion_occupancy_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, Volume& vol);



struct TsdfMaskFusionFunctor : public FusionFunctor {
  float truncation_;
  bool unknown_is_free_;
  TsdfMaskFusionFunctor(float truncation, bool unknown_is_free) :
    truncation_(truncation), unknown_is_free_(unknown_is_free) {}

  FUSION_FUNCTION
  virtual bool new_sample(Volume* vol, float vx_depth, float dm_depth, int d, int h, int w, int* n_valid_views) const {
    if(unknown_is_free_ && dm_depth < 0) {
      dm_depth = 1e9;
    }
    float diff = dm_depth - vx_depth;
    if(dm_depth > 0 && diff >= -truncation_) {
      volume_set(vol, 0,d,h,w, 1);
      return false;
    }
    return true;
  }
};

void fusion_tsdfmask_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, Volume& vol);


struct TsdfFusionFunctor : public FusionFunctor {
  float truncation_;
  bool unknown_is_free_;
  TsdfFusionFunctor(float truncation, bool unknown_is_free) :
    truncation_(truncation), unknown_is_free_(unknown_is_free) {}

  FUSION_FUNCTION
  virtual void before_sample(Volume* vol, int d, int h, int w) const {
    for(int c = 0; c < vol->channels_; ++c) {
      volume_set(vol, c,d,h,w, 0);
    }
  }

  FUSION_FUNCTION
  virtual bool new_sample(Volume* vol, float vx_depth, float dm_depth, int d, int h, int w, int* n_valid_views) const {
    if(unknown_is_free_ && dm_depth < 0) {
      dm_depth = 1e9;
    }
    float dist = dm_depth - vx_depth;
    float truncated_dist = fminf(truncation_, fmaxf(-truncation_, dist));
    if(dm_depth > 0 && dist >= -truncation_) {
      (*n_valid_views)++;
      volume_add(vol, 0,d,h,w, truncated_dist);
    }
    return true;
  }

  FUSION_FUNCTION 
  virtual void after_sample(Volume* vol, int d, int h, int w, int n_valid_views) const {
	  if(n_valid_views > 0) {
      volume_div(vol, 0,d,h,w, n_valid_views);
    } 
    else {
      volume_set(vol, 0,d,h,w, -truncation_);
    }
  }
};

void fusion_tsdf_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, Volume& vol);


struct TsdfHistFusionFunctor : public FusionFunctor {
  float truncation_;
  bool unknown_is_free_;
  float* bin_centers_;
  int n_bins_;
  bool unobserved_is_occupied_;
  TsdfHistFusionFunctor(float truncation, bool unknown_is_free, float* bin_centers, int n_bins, bool unobserved_is_occupied) :
    truncation_(truncation), unknown_is_free_(unknown_is_free), bin_centers_(bin_centers), n_bins_(n_bins), unobserved_is_occupied_(unobserved_is_occupied) {}

  FUSION_FUNCTION
  virtual bool new_sample(Volume* vol, float vx_depth, float dm_depth, int d, int h, int w, int* n_valid_views) const {
    if(unknown_is_free_ && dm_depth < 0) {
      dm_depth = 1e9;
    }
    float dist = dm_depth - vx_depth;
    
    if(dm_depth > 0 && dist >= -truncation_) {
      (*n_valid_views)++;
      if(dist <= bin_centers_[0]) {
        volume_add(vol, 0,d,h,w, 1);
      }
      else if(dist >= bin_centers_[n_bins_-1]) {
        volume_add(vol, n_bins_-1,d,h,w, 1);
      }
      else {
        int bin = 0;
        while(dist > bin_centers_[bin]) {
          bin++;
        }
        float a = fabs(bin_centers_[bin-1] - dist);
        float b = fabs(bin_centers_[bin] - dist);
        volume_add(vol, bin-1,d,h,w, a / (a+b));
        volume_add(vol, bin,  d,h,w, b / (a+b));
      }
    }
    return true;
  }

  FUSION_FUNCTION 
  virtual void after_sample(Volume* vol, int d, int h, int w, int n_valid_views) const {
	  if(n_valid_views > 0) {
      for(int bin = 0; bin < n_bins_; ++bin) {
        volume_div(vol, bin,d,h,w, n_valid_views);
      }
    } 
    else if(unobserved_is_occupied_) {
      volume_set(vol, 0,d,h,w, 1);
    }
  }
};

void fusion_tsdf_hist_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, float* bin_centers, int n_bins, bool unobserved_is_occupied, Volume& vol);




void fusion_hist_zach_tvl1_gpu(const Volume& hist, bool hist_on_gpu, float truncation, float lambda, int iterations, Volume& vol);
void fusion_zach_tvl1_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, float* bin_centers, int n_bins, float lambda, int iterations, Volume& vol);

#endif
