#ifndef __DMC_CUDA_KERNELS__
#define __DMC_CUDA_KERNELS__

#include <torch/torch.h>
#include <vector>


// Curvature constraint
void curvature_constraint_kernel_forward(
    at::Tensor offset,
    at::Tensor topology,
    at::Tensor xTable,
    at::Tensor yTable,
    at::Tensor zTable,
    at::Tensor innerTable,
    at::Tensor loss_x,
    at::Tensor loss_y,
    at::Tensor loss_z,
    at::Tensor loss_inner);


void curvature_constraint_kernel_backward(
  at::Tensor grad_output, 
  at::Tensor offset,
  at::Tensor topology,
  at::Tensor xTable,
  at::Tensor yTable,
  at::Tensor zTable,
  at::Tensor innerTable,
  at::Tensor grad_offset);

  // Grid pooling
void grid_pooling_kernel_forward( 
    at::Tensor point, 
    at::Tensor feat_points,
    at::Tensor shape,
    at::Tensor feat_cell,  
    at::Tensor indices);

void grid_pooling_kernel_backward( 
    at::Tensor grad_output,
    at::Tensor shape, 
    at::Tensor indices,
    at::Tensor grad_feat_points);

// Occ2Topo
void occupancy_to_topology_kernel_forward( 
    at::Tensor occupancy, 
    at::Tensor topology );

void occupancy_to_topology_kernel_backward( 
    at::Tensor grad_output,
    at::Tensor occupancy, 
    at::Tensor topology,
    at::Tensor grad_occupancy);

// OccConstraint
void occupancy_connectivity_kernel_forward( 
    at::Tensor occupancy,
    at::Tensor loss);

void occupancy_connectivity_kernel_backward(
    at::Tensor grad_output, 
    at::Tensor occupancy, 
    at::Tensor grad_occupancy);

// Points Triangle distance
void point_topology_distance_kernel_forward( 
  at::Tensor offset,
  at::Tensor points,
  at::Tensor distances,
  at::Tensor indices_all);

void point_topology_distance_kernel_backward( 
  at::Tensor grad_output,
  at::Tensor offset,
  at::Tensor points,
  at::Tensor indices_all,
  at::Tensor grad_offset);

#endif