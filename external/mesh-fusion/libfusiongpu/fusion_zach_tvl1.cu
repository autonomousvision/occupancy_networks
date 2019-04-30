#include "gpu_common.h"

__global__ void kernel_zach_tvl1_dual(Volume u, Volume p, Volume hist, int vx_res3, float sigma) {
  CUDA_KERNEL_LOOP(idx, vx_res3) {
    int d,h,w;
    fusion_idx2dhw(idx, u.width_,u.height_, d,h,w);

    float u_curr = u.data_[idx];

    float u_x = u.data_[idx + (w < u.width_-1)] - u_curr;
    float u_y = u.data_[idx + (h < u.height_-1) * u.width_] - u_curr;
    float u_z = u.data_[idx + (d < u.depth_-1) * u.width_ * u.height_] - u_curr;

    float p0_new = p.data_[idx + 0 * vx_res3] + sigma * u_x;
    float p1_new = p.data_[idx + 1 * vx_res3] + sigma * u_y;
    float p2_new = p.data_[idx + 2 * vx_res3] + sigma * u_z;

    float denom = fmax(1.0f, sqrtf(p0_new*p0_new + p1_new*p1_new + p2_new*p2_new));

    p.data_[idx + 0 * vx_res3] = p0_new / denom;
    p.data_[idx + 1 * vx_res3] = p1_new / denom;
    p.data_[idx + 2 * vx_res3] = p2_new / denom;
  }
}

__global__ void kernel_zach_tvl1_primal(Volume u, Volume p, Volume hist, int vx_res3, float tau, float lambda) {
  CUDA_KERNEL_LOOP(idx, vx_res3) {
    int d,h,w;
    fusion_idx2dhw(idx, u.width_,u.height_, d,h,w);
    
    float px = (w > 0) * p.data_[idx + 0 * vx_res3 - (w>0)] - p.data_[idx + 0 * vx_res3];
    float py = (h > 0) * p.data_[idx + 1 * vx_res3 - (h>0) * u.width_] - p.data_[idx + 1 * vx_res3];
    float pz = (d > 0) * p.data_[idx + 2 * vx_res3 - (d>0) * u.width_ * u.height_] - p.data_[idx + 2 * vx_res3];

    float u_old = u.data_[idx];

    float divergence = px + py + pz;
    float u_new = u_old - tau * divergence;


    int n_bins = hist.channels_;
    extern __shared__ float shared[];
    float* arr_w = shared + threadIdx.x * 2 * (n_bins + 1);
    float* arr_W = arr_w + (n_bins + 1);
    float* arr_l = arr_w;
    for(int i = 0; i < n_bins; ++i) {
      arr_w[i] = hist.data_[i * vx_res3 + idx];
    }

    for(int i = 0; i <= n_bins; ++i) {
      arr_W[i] = 0;
      for(int j = 1; j <= i; ++j) {
        arr_W[i] -= arr_w[j-1];
      }
      for(int j = i+1; j <= n_bins; ++j) {
        arr_W[i] += arr_w[j-1];
      }
    }

    for(int i = 0; i < n_bins; ++i) {
      arr_l[i] = ((2.0f * i) / (n_bins - 1.0f)) - 1.0f;
    }
    arr_l[n_bins] = 1e9;


    for(int i = 0; i <= n_bins; ++i) {
      float p = u_new + tau * lambda * arr_W[i];
      for (int j = n_bins; j >= 0; j--) {
        if (p < arr_l[j]) {
          float tmp = arr_l[j];
          arr_l[j] = p;
          if (j < n_bins) {
            arr_l[j+1] = tmp;
          }
        } else {
          break;
        }
      }
    }

    u.data_[idx] = fminf(1.0f, fmaxf(-1.0f, arr_l[n_bins]));
  }
}



void fusion_hist_zach_tvl1_gpu(const Volume& hist, bool hist_on_gpu, float truncation, float lambda, int iterations, Volume& vol) {
  Volume hist_gpu;
  if(hist_on_gpu) {
    hist_gpu = hist;
  }
  else {
    volume_to_gpu(hist, hist_gpu, true);
  }
  int n_bins = hist.channels_;
  int vx_res3 = hist.depth_ * hist.height_ * hist.width_;


  // primal-dual algorithm
  Volume u, p;
  volume_to_gpu(vol, u, true); 
  volume_alloc_gpu(3, vol.depth_,vol.height_,vol.width_, p);
  volume_fill_data_gpu(p, 0);

  float tau = 1.0/sqrt(6.0f)/3;
  float sigma = 1.0/sqrt(6.0f)*3;

  for(int iter = 0; iter < iterations; ++iter) {
    if((iter+1) % 100 == 0) {
      printf("  zach_tvl1 iter=% 4d\n", iter+1);
    }
    kernel_zach_tvl1_dual<<<GET_BLOCKS(vx_res3), CUDA_NUM_THREADS>>>(
      u, p, hist_gpu, vx_res3, sigma
    );
    kernel_zach_tvl1_primal<<<GET_BLOCKS_T(vx_res3, 256), 256, 256 * 2 * (n_bins+1) * sizeof(float)>>>(
      u, p, hist_gpu, vx_res3, tau, lambda
    );
    cudaDeviceSynchronize();
  }
  CUDA_POST_KERNEL_CHECK;

  volume_mul_data_gpu(u, truncation);

  volume_to_cpu(u, vol, false);

  volume_free_gpu(u);
  volume_free_gpu(p);


  if(!hist_on_gpu) {
    volume_free_gpu(hist_gpu);
  }
}

void fusion_zach_tvl1_gpu(const Views& views, float vx_size, float truncation, bool unknown_is_free, float* bin_centers, int n_bins, float lambda, int iterations, Volume& vol) {
  //compute hist
  Views views_gpu;
  views_to_gpu(views, views_gpu, true);

  Volume hist;
  float* bin_centers_gpu = host_to_device_malloc(bin_centers, n_bins);
  volume_alloc_gpu(n_bins, vol.depth_, vol.height_, vol.width_, hist);
  bool unobserved_is_occupied = true;
  TsdfHistFusionFunctor functor(truncation, unknown_is_free, bin_centers_gpu, n_bins, unobserved_is_occupied);
  device_free(bin_centers_gpu);

  int vx_res3 = vol.depth_ * vol.height_ * vol.width_;
  kernel_fusion<<<GET_BLOCKS(vx_res3), CUDA_NUM_THREADS>>>(
    vx_res3, views_gpu, functor, vx_size, hist
  );
  CUDA_POST_KERNEL_CHECK;
  
  views_free_gpu(views_gpu);
  
  fusion_hist_zach_tvl1_gpu(hist, true, truncation, lambda, iterations, vol);
}
