#include <ATen/ATen.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
__constant__ float grid_size=1.0;
namespace{
/**
 * perform max-pooling within the cells
 * parallel over each cell and each feature dimension
 */
 template <typename scalar_t>
 __global__ void grid_pooling_kernel( 
      const scalar_t *point, 
      const scalar_t *feat_points, 
      scalar_t *feat_cell, 
      int *indices, 
      const int n  ){
  // cell indices
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;
  // cell size
  // int W = gridDim.x;
  int H = gridDim.y;
  int D = gridDim.z;
  int ind = i*H*D + j*D + k;
  int c = threadIdx.x;
  int C = blockDim.x;
  
  for (int p=0; p<n; p++){
    scalar_t px = point[p*3+0];
    scalar_t py = point[p*3+1];
    scalar_t pz = point[p*3+2];
     // if point is inside of the grid
    if (px >= i && px < i+grid_size && py >= j && py < j+grid_size && pz >= k && pz < k+grid_size){
    // max-pooling, update feat_cell if the feature is larger than the current feat_cell
    // can be async for max operation
    if ( feat_points[p*C + c] > feat_cell[ind*C + c] ){
       feat_cell[ind*C + c] = feat_points[p*C + c];
       indices[ind*C + c] = p;
    }
     }
  }
}
/**
 * back-propagate the loss from the max-pooled feature to point features
 * parallel over each cell and each feature dimension
 */
template <typename scalar_t>
 __global__ void grad_grid_pooling_kernel( 
      const scalar_t *grad_output, 
      const int *indices, 
      scalar_t *grad_feat_points){
  // cell indices
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;
  // cell size
  // int W = gridDim.x;
  int H = gridDim.y;
  int D = gridDim.z;
  int ind = i*H*D + j*D + k;
  int c = threadIdx.x;
  int C = blockDim.x;
  long int p = indices[ind*C + c];
  if (p < 0) return;

  grad_feat_points[p*C + c] = grad_output[ind*C + c];
}
} //namespace

/*
 * Forward function, project the point features to cells, perform max pooling in every cell 
 * params: 
 *  	  point        input, all points, Nx3
 *  	  feat_points  input, feature of all points, NxC
 *  	  shape 	   input, size of the grid [W, H, D], 3
 *  	  feat_cell    output, feature of all cells, (WxHxD)xC
 *  	  indices      output, indices of max pooling, saved for back propagation, (WxHxD)xC
 *
 */	

void grid_pooling_kernel_forward( 
    at::Tensor point, 
    at::Tensor feat_points,
    at::Tensor shape,
    at::Tensor feat_cell,  
    at::Tensor indices){
  int W = shape.data<long>()[0];
  int H = shape.data<long>()[1];
  int D = shape.data<long>()[2];
  int C = feat_cell.size(1);

  dim3 dimGrid(W, H, D);
  dim3 dimBlock(C, 1, 1);
  // lauch the kernel
  int n = point.size(0);
  grid_pooling_kernel<float><<< dimGrid, dimBlock>>>(
      point.data<float>(),
      feat_points.data<float>(),
      feat_cell.data<float>(),
      indices.data<int>(),
      n);
}

/*
 * Backward function, back-propagate the loss to the point features
 * params: 
 *    grad_output   	input, gradient on the output feature, WxHxC 
 *    shape 		    input, size of the grid [W, H, D], 3
 *    indices     		input, indices of max pooling, WxHxC
 * 	  grad_feat_points 	output, gradient on the features, NxC 
 *
*/
void grid_pooling_kernel_backward( 
    at::Tensor grad_output,
    at::Tensor shape, 
    at::Tensor indices,
    at::Tensor grad_feat_points){
  int W = shape.data<long>()[0];
  int H = shape.data<long>()[1];
  int D = shape.data<long>()[2];
  int C = grad_output.size(1);
  dim3 dimGrid(W, H, D);
  dim3 dimBlock(C, 1, 1);
  // lauch the kernel
  grad_grid_pooling_kernel<float><<< dimGrid, dimBlock>>>(
      grad_output.data<float>(),
      indices.data<int>(),
      grad_feat_points.data<float>());
}
    

