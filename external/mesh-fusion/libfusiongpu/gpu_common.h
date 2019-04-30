#include "fusion.h"

#include <cublas_v2.h>

#include <thrust/execution_policy.h>
// #include <thrust/transform_reduce.h>
// #include <thrust/transform_scan.h>
// #include <thrust/iterator/counting_iterator.h>
// #include <thrust/functional.h>
// #include <thrust/equal.h>


#define DEBUG 0
#define CUDA_DEBUG_DEVICE_SYNC 0

inline void getCudaMemUsage(size_t* free_byte, size_t* total_byte) {
  cudaError_t stat = cudaMemGetInfo(free_byte, total_byte);
  if(stat != cudaSuccess) {
    printf("[ERROR] failed to query cuda memory (%f,%f) information, %s\n", *free_byte/1024.0/1024.0, *total_byte/1024.0/1024.0, cudaGetErrorString(stat));
    exit(-1);
  }
}

inline void printCudaMemUsage() {
  size_t free_byte, total_byte;
  getCudaMemUsage(&free_byte, &total_byte);
  printf("[INFO] CUDA MEM Free=%f[MB], Used=%f[MB]\n", free_byte/1024.0/1024.0, total_byte/1024.0/1024.0);
}

// cuda check for cudaMalloc and so on
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    if(CUDA_DEBUG_DEVICE_SYNC) { cudaDeviceSynchronize(); } \
    cudaError_t error = condition; \
    if(error != cudaSuccess) { \
      printf("%s in %s at %d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
      exit(-1); \
    } \
  } while (0)


inline const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "Unknown cublas status";
}

#define CUBLAS_CHECK(condition) \
  do { \
    if(CUDA_DEBUG_DEVICE_SYNC) { cudaDeviceSynchronize(); } \
    cublasStatus_t status = condition; \
    if(status != CUBLAS_STATUS_SUCCESS) { \
      printf("%s in %s at %d\n", cublasGetErrorString(status), __FILE__, __LINE__); \
      exit(-1); \
    } \
  } while (0)


// check if there is a error after kernel execution
#define CUDA_POST_KERNEL_CHECK \
  CUDA_CHECK(cudaPeekAtLastError()); \
  CUDA_CHECK(cudaGetLastError()); 

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS_T(const int N, const int N_THREADS) {
  return (N + N_THREADS - 1) / N_THREADS;
}
inline int GET_BLOCKS(const int N) {
  return GET_BLOCKS_T(N, CUDA_NUM_THREADS);
  // return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template<typename T>
T* device_malloc(long N) {
  T* dptr;
  CUDA_CHECK(cudaMalloc(&dptr, N * sizeof(T)));
  if(DEBUG) { printf("[DEBUG] device_malloc %p, %ld\n", dptr, N); }
  return dptr;
}

template<typename T>
void device_free(T* dptr) {
  if(DEBUG) { printf("[DEBUG] device_free %p\n", dptr); }
  CUDA_CHECK(cudaFree(dptr));
}

template<typename T>
void host_to_device(const T* hptr, T* dptr, long N) {
  if(DEBUG) { printf("[DEBUG] host_to_device %p => %p, %ld\n", hptr, dptr, N); }
  CUDA_CHECK(cudaMemcpy(dptr, hptr, N * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
T* host_to_device_malloc(const T* hptr, long N) {
  T* dptr = device_malloc<T>(N);
  host_to_device(hptr, dptr, N);
  return dptr;
}

template<typename T>
void device_to_host(const T* dptr, T* hptr, long N) {
  if(DEBUG) { printf("[DEBUG] device_to_host %p => %p, %ld\n", dptr, hptr, N); }
  CUDA_CHECK(cudaMemcpy(hptr, dptr, N * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
T* device_to_host_malloc(const T* dptr, long N) {
  T* hptr = new T[N];
  device_to_host(dptr, hptr, N);
  return hptr;
}

template<typename T>
void device_to_device(const T* dptr, T* hptr, long N) {
  if(DEBUG) { printf("[DEBUG] device_to_device %p => %p, %ld\n", dptr, hptr, N); }
  CUDA_CHECK(cudaMemcpy(hptr, dptr, N * sizeof(T), cudaMemcpyDeviceToDevice));
}



inline void views_to_gpu(const Views& views_cpu, Views& views_gpu, bool alloc) {
  views_gpu.n_views_ = views_cpu.n_views_;
  views_gpu.rows_ = views_cpu.rows_;
  views_gpu.cols_ = views_cpu.cols_;

  printf("    views_to_gpu with %dx%dx%d\n", views_gpu.n_views_, views_gpu.rows_, views_gpu.cols_);
  int N = views_cpu.n_views_ * views_cpu.rows_ * views_cpu.cols_;
  if(alloc) {
    views_gpu.depthmaps_ = device_malloc<float>(N);
  }
  host_to_device(views_cpu.depthmaps_, views_gpu.depthmaps_, N);

  N = views_cpu.n_views_ * 3 * 3;
  if(alloc) {
    views_gpu.Ks_ = device_malloc<float>(N);
  }
  host_to_device(views_cpu.Ks_, views_gpu.Ks_, N);

  N = views_cpu.n_views_ * 3 * 3;
  if(alloc) {
    views_gpu.Rs_ = device_malloc<float>(N);
  }
  host_to_device(views_cpu.Rs_, views_gpu.Rs_, N);

  N = views_cpu.n_views_ * 3;
  if(alloc) {
    views_gpu.Ts_ = device_malloc<float>(N);
  }
  host_to_device(views_cpu.Ts_, views_gpu.Ts_ , N);
}

inline void views_free_gpu(Views& views_gpu) {
  device_free(views_gpu.depthmaps_);
  device_free(views_gpu.Ks_);
  device_free(views_gpu.Rs_);
  device_free(views_gpu.Ts_);
}



inline void volume_alloc_like_gpu(const Volume& vol_cpu, Volume& vol_gpu) {
  vol_gpu.channels_ = vol_cpu.channels_;
  vol_gpu.depth_ = vol_cpu.depth_;
  vol_gpu.height_ = vol_cpu.height_;
  vol_gpu.width_ = vol_cpu.width_;
  int N = vol_cpu.channels_ * vol_cpu.depth_ * vol_cpu.height_ * vol_cpu.width_;
  printf("  volume_alloc_like_gpu gpu memory for volume %dx%dx%dx%d\n", vol_gpu.channels_, vol_gpu.depth_, vol_gpu.height_, vol_gpu.width_);
  vol_gpu.data_ = device_malloc<float>(N);
}

inline void volume_alloc_gpu(int channels, int depth, int height, int width, Volume& vol_gpu) {
  vol_gpu.channels_ = channels;
  vol_gpu.depth_ = depth;
  vol_gpu.height_ = height;
  vol_gpu.width_ = width;
  int N = channels * depth * height * width;
  printf("  volume_alloc_gpu gpu memory for volume %dx%dx%dx%d\n", vol_gpu.channels_, vol_gpu.depth_, vol_gpu.height_, vol_gpu.width_);
  vol_gpu.data_ = device_malloc<float>(N);
}

inline void volume_fill_data_gpu(Volume& vol, float fill_value) {
  int n = vol.channels_ * vol.depth_ * vol.height_ * vol.width_;
  thrust::fill_n(thrust::device, vol.data_, n, fill_value);
}

template<class T>
struct scalar_multiply {
  T s;
  scalar_multiply(T _s) : s(_s) {}

  __host__ __device__ T operator()(T& x) const {
    return x * s;
  }
};

inline void volume_mul_data_gpu(Volume& vol, float val) {
  int n = vol.channels_ * vol.depth_ * vol.height_ * vol.width_;
  thrust::transform(thrust::device, vol.data_, vol.data_ + n, vol.data_, scalar_multiply<float>(val));
}

inline void volume_to_gpu(const Volume& vol_cpu, Volume& vol_gpu, bool alloc) {
  vol_gpu.channels_ = vol_cpu.channels_;
  vol_gpu.depth_ = vol_cpu.depth_;
  vol_gpu.height_ = vol_cpu.height_;
  vol_gpu.width_ = vol_cpu.width_;

  int N = vol_cpu.channels_ * vol_cpu.depth_ * vol_cpu.height_ * vol_cpu.width_;
  if(alloc) {
    vol_gpu.data_ = device_malloc<float>(N);
  }
  host_to_device(vol_cpu.data_, vol_gpu.data_, N);
}

inline void volume_to_cpu(const Volume& vol_gpu, Volume& vol_cpu, bool alloc) {
  vol_cpu.channels_ = vol_gpu.channels_;
  vol_cpu.depth_ = vol_gpu.depth_;
  vol_cpu.height_ = vol_gpu.height_;
  vol_cpu.width_ = vol_gpu.width_;

  int N = vol_gpu.channels_ * vol_gpu.depth_ * vol_gpu.height_ * vol_gpu.width_;
  if(alloc) {
    vol_cpu.data_ = new float[N];
  }
  device_to_host(vol_gpu.data_, vol_cpu.data_, N);
}

inline void volume_free_gpu(Volume& vol_gpu) {
  device_free(vol_gpu.data_);
}




template <typename FusionFunctorT>
__global__ void kernel_fusion(int vx_res3, const Views views, const FusionFunctorT functor, float vx_size, Volume vol);
