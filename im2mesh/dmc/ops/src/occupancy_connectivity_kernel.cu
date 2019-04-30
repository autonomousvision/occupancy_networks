#include <ATen/ATen.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
__constant__ float grid_size=1.0;
namespace{

/**
 * calculate the loss between two neigboring occupancy status 
 */	
 template <typename scalar_t>
 __global__ void occupancy_connectivity_kernel( const scalar_t *occupancy, scalar_t *loss ){

  int i=blockIdx.x;
  int j=blockIdx.y;
  int k=blockIdx.z;

  int W=gridDim.x-1;
  int H=gridDim.y-1;
  int D=gridDim.z-1;

  scalar_t loss_ = 0.0;

  scalar_t p1 = occupancy[ i*(H+1)*(D+1) + j*(D+1) + k ]; 
 
  if (j<H){
    scalar_t p2 = occupancy[ i*(H+1)*(D+1) + (j+1)*(D+1) + k ]; 
      // l1 loss
      loss_ += abs(p1-p2);
  }
  if (i<W){
    scalar_t p3 = occupancy[ (i+1)*(H+1)*(D+1) + j*(D+1) + k ]; 
      // l1 loss
      loss_ += abs(p1-p3);
  }
  if (k<D){
    scalar_t p4 = occupancy[ i*(H+1)*(D+1) + j*(D+1) + k+1 ]; 
      // l1 loss
      loss_ += abs(p1-p4);
  }
  loss[ i*(H+1)*(D+1) + j*(D+1) + k ] = loss_;
}


/**
 * propagate the gradient to the occupancy status 
 */	
 template <typename scalar_t>
 __global__ void grad_occupancy_connectivity_kernel( const scalar_t *occupancy, scalar_t *grad_occupancy ){

  int i=blockIdx.x;
  int j=blockIdx.y;
  int k=blockIdx.z;

  int W=gridDim.x-1;
  int H=gridDim.y-1;
  int D=gridDim.z-1;

  scalar_t p1 = occupancy[ i*(H+1)*(D+1) + j*(D+1) + k ]; 
 
  if (j<H){
      scalar_t p2 = occupancy[ i*(H+1)*(D+1) + (j+1)*(D+1) + k ]; 
      // l1 loss
      scalar_t sign;
      if (p1>p2){ sign = 1.0; }else{ sign = -1.0; }
      atomicAdd( &grad_occupancy[ i*(H+1)*(D+1) + j*(D+1) + k ], sign );
      atomicAdd( &grad_occupancy[ i*(H+1)*(D+1) + (j+1)*(D+1) + k ], -sign );

  }
  if (i<W){
      scalar_t p3 = occupancy[ (i+1)*(H+1)*(D+1) + j*(D+1) + k ]; 
      // l1 loss
      scalar_t sign;
      if (p1>p3){ sign = 1.0; }else{ sign = -1.0; }
      atomicAdd( &grad_occupancy[ i*(H+1)*(D+1) + j*(D+1) + k ], sign );
      atomicAdd( &grad_occupancy[ (i+1)*(H+1)*(D+1) + j*(D+1) + k ], -sign );
  }
  if (k<D){
      scalar_t p4 = occupancy[ i*(H+1)*(D+1) + j*(D+1) + k+1 ]; 
      scalar_t sign;
      if (p1>p4){ sign = 1.0; }else{ sign = -1.0; }
      atomicAdd( &grad_occupancy[ i*(H+1)*(D+1) + j*(D+1) + k ], sign );
      atomicAdd( &grad_occupancy[ i*(H+1)*(D+1) + j*(D+1) + k+1 ], -sign );
  }
}
} //namespace

/*
 * Forward function, regularize the neighboring occupancy status to be close 
 * params: 
 *  	  occupancy 	input, (W+1)x(H+1)x(D+1)
 *  	  loss      	output, connectivity loss 
 *
 */
void occupancy_connectivity_kernel_forward( 
      at::Tensor occupancy,
      at::Tensor loss_all){
  int W = occupancy.size(0);
  int H = occupancy.size(1);
  int D = occupancy.size(2);
      
  dim3 dimGrid(W, H, D);
  // lauch the kernel
  assert(occupancy.type().scalarType() == at::ScalarType::Float);
  assert(loss_all.type().scalarType() == at::ScalarType::Float);
  
  occupancy_connectivity_kernel<float><<< dimGrid, 1>>>(
      occupancy.data<float>(),
      loss_all.data<float>());
}

/*
 * Backward function, propagate the loss to every occupancy status 
 * params: 
 *  	  grad_output 		input, 1, gradient on the loss 
 *  	  occupancy 	  	input, (W+1)x(H+1)x(D+1)
 *  	  grad_occupancy 	output, (W+1)x(H+1)x(D+1), gradient on the occupancy 
 *
 */
void occupancy_connectivity_kernel_backward(
      at::Tensor grad_output, 
      at::Tensor occupancy, 
      at::Tensor grad_occupancy){
    int W = occupancy.size(0);
    int H = occupancy.size(1);
    int D = occupancy.size(2);

    dim3 dimGrid(W, H, D);
    assert(occupancy.type().scalarType() == at::ScalarType::Float);
    assert(grad_output.type().scalarType() == at::ScalarType::Float);
    assert(grad_occupancy.type().scalarType() == at::ScalarType::Float);
    grad_occupancy_connectivity_kernel<float><<< dimGrid, 1>>>(
        occupancy.data<float>(),
        grad_occupancy.data<float>());
    // Multiply with incoming gradient
    // -> do that in Python now
    // grad_occupancy *= grad_output;
}
  

