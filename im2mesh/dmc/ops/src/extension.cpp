#include "kernels.h"

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Curvature Constraint
/*
 * Forward function, calculating the distance from a set of points to one single linesegment 
 * params: 
 * 	  offset 		input, offset map for x,y directions, 2x(W+1)x(H+1)x(D+1) 
 * 	  topolopy 		input, probability for each triangle, (WxHxD)xT'
 * 	  topology_empty 	input, probability for empty topology, (WxHxD) 
 *    xTable 	        input, connected __triangles__ in x direction, T'xT' (T'>=T) 
 *    yTable 	        input, connected __triangles__ in y direction, T'xT' (T'>=T)
 *    zTable 	        input, connected __triangles__ in z direction, T'xT' (T'>=T)
 *    innerTable 	    input, connected __triangles__ for one topology within a single cell, T'xT' (T'>=T)
 *    loss  		    output, smoothness loss on both horizontal and vertical directions, 1 
 *
 */	
at::Tensor curvature_constraint_cuda_forward( 
    at::Tensor offset,
    at::Tensor topology,
    at::Tensor xTable,
    at::Tensor yTable,
    at::Tensor zTable,
    at::Tensor innerTable) {
  CHECK_INPUT(offset);
  CHECK_INPUT(topology);
  CHECK_INPUT(xTable);
  CHECK_INPUT(yTable);
  CHECK_INPUT(zTable);
  CHECK_INPUT(innerTable);

  // Declare losses
  int W = offset.size(1) - 1;
  int H = offset.size(2) - 1;
  int D = offset.size(3) - 1;

  auto loss_x = at::zeros({W * H * D}, torch::CUDA(at::kFloat));
  auto loss_y = at::zeros({W * H * D}, torch::CUDA(at::kFloat));
  auto loss_z =  at::zeros({W * H * D}, torch::CUDA(at::kFloat));
  auto loss_inner = at::zeros({W * H * D}, torch::CUDA(at::kFloat));


  // Declare losses
  curvature_constraint_kernel_forward(
      offset, topology, xTable, yTable, zTable, innerTable,
      loss_x, loss_y, loss_z, loss_inner);

  auto loss = loss_x.sum() + loss_y.sum() + loss_z.sum() + loss_inner.sum();
  return loss;
}

/*
 * Backward function, calculating the derivative of the topology with respect to the loss 
 * params: 
 * 	  grad_output 		input, gradient on the output loss, 1
 * 	  offset 		input, offset map for x,y directions, 3x(W+1)x(H+1)x(D+1) 
 * 	  topolopy 		input, probability for each triangle, (WxHxD)xT'
 * 	  topology_empty 	input, probability for empty topology, (WxHxD) 
 *    xTable 	        input, connected __triangles__ in x direction, T'xT' (T'>=T) 
 *    yTable 	        input, connected __triangles__ in y direction, T'xT' (T'>=T)
 *    zTable 	        input, connected __triangles__ in z direction, T'xT' (T'>=T)
 *    innerTable 	        input, connected __triangles__ for one topology within a single cell, T'xT' (T'>=T)
 *    grad_offset  		output, gradient on the offset, 3x(W+1)x(H+1)x(D+1) 
 *
 */
void curvature_constraint_cuda_backward(
    at::Tensor grad_output, 
    at::Tensor offset,
    at::Tensor topology,
    at::Tensor xTable,
    at::Tensor yTable,
    at::Tensor zTable,
    at::Tensor innerTable,
    at::Tensor grad_offset) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(offset);
  CHECK_INPUT(topology);
  CHECK_INPUT(xTable);
  CHECK_INPUT(yTable);
  CHECK_INPUT(zTable);
  CHECK_INPUT(innerTable);
  CHECK_INPUT(grad_offset);

  curvature_constraint_kernel_backward(grad_output, offset, topology, xTable, yTable, zTable, innerTable, grad_offset);
}


// Grid pooling
/*
 * Forward function, project the point features to cells, perform max pooling in every cell 
 * params: 
 *     point 	    input, all points, Nx2
 *     feat_points  input, feature of all points, NxC
 *     shape 	    input, size of the grid [W, H, D], 3
 *     feat_cell    output, feature of all cells, (WxHxD)xC
 *     indices     	output, indices of max pooling, saved for back propagation, (WxHxD)xC
 *
 */	
void grid_pooling_cuda_forward( 
  at::Tensor point, 
  at::Tensor feat_points, 
  at::Tensor shape,
  at::Tensor feat_cell,  
  at::Tensor indices) {
  CHECK_INPUT(point);
  CHECK_INPUT(feat_points);
  CHECK_CONTIGUOUS(shape);
  CHECK_INPUT(feat_cell);
  CHECK_INPUT(indices);
  
  grid_pooling_kernel_forward(point, feat_points, shape, feat_cell, indices);
}

/*
 * Backward function, back-propagate the loss to the point features
 * params: 
 *     grad_output   	input, gradient on the output feature, WxHxC 
 *     shape 		    input, size of the grid [W, H, D], 3
 *     indices     		input, indices of max pooling, WxHxC
 * 	   grad_feat_points output, gradient on the features, NxC 
 *
 */	
void grid_pooling_cuda_backward( 
  at::Tensor grad_output, 
  at::Tensor shape,  
  at::Tensor indices,
  at::Tensor grad_feat_points) {
  CHECK_INPUT(grad_output);
  CHECK_CONTIGUOUS(shape);
  CHECK_INPUT(indices);
  CHECK_INPUT(grad_feat_points);

  grid_pooling_kernel_backward(grad_output, shape, indices, grad_feat_points);
}

// Occupancy Connectivity
/*
 * Forward function, regularize the neighboring occupancy status to be close 
 * params: 
 *     occupancy input, (W+1)x(H+1)x(D+1)
 *     loss     	output, connectivity loss 
 *
 */	
at::Tensor occupancy_connectivity_cuda_forward( 
    at::Tensor occupancy) {
  CHECK_INPUT(occupancy);
  int W = occupancy.size(0);
  int H = occupancy.size(1);
  int D = occupancy.size(2);

  auto loss_all = at::zeros({W*H*D}, torch::CUDA(at::kFloat));
  occupancy_connectivity_kernel_forward(occupancy, loss_all);
  return loss_all.sum();
}

/*
 * Backward function, propagate the loss to every occupancy status 
 * params: 
 *    grad_output 		input, 1, gradient on the loss 
 *    occupancy 		input, (W+1)x(H+1)x(D+1)
 *    grad_occupancy   	output, (W+1)x(H+1)x(D+1), gradient on the occupancy 
 *
 */
void occupancy_connectivity_cuda_backward( 
  at::Tensor grad_output, 
  at::Tensor occupancy, 
  at::Tensor grad_occupancy){
  CHECK_INPUT(grad_output);
  CHECK_INPUT(occupancy);
  CHECK_INPUT(grad_occupancy);

  occupancy_connectivity_kernel_backward(grad_output ,occupancy, grad_occupancy);
}


// Occupancy To Topology
/*
 * Forward function, compute the topology probability given the occupancy probability 
 * params: 
 *      occupancy 	input, (W+1)x(H+1)x(D+1)
 *      topology    output, probability of all topologies types we care about (WxHxD)xT
 *
 */	

void occupancy_to_topology_cuda_forward(
    at::Tensor occupancy,
    at::Tensor topology) {
  CHECK_INPUT(occupancy);
  CHECK_INPUT(topology);

  occupancy_to_topology_kernel_forward(occupancy, topology);
}

/*
 * Backward function, backpropagate the gradient from topology to occupancy 
 * params: 
 * 	  grad_output   	input, gradient on the topology probability, (WxHxD)xT
 *    occupancy 		input, (W+1)x(H+1)x(D+1)
 *    topology     		input, probability of all topologies types we care about (WxHxD)xT
 *    grad_occupancy   	output, gradient on the occupancy map, (W+1)x(H+1)x(D+1) 
 *
 */
void occupancy_to_topology_cuda_backward(
  at::Tensor grad_output,
  at::Tensor occupancy, 
  at::Tensor topology,
  at::Tensor grad_occupancy) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(occupancy);
  CHECK_INPUT(topology);
  CHECK_INPUT(grad_occupancy);

  occupancy_to_topology_kernel_backward(grad_output, occupancy, topology, grad_occupancy);
}


// Point Triangle distance
/*
 * Forward function, calculating the point to mesh distances for all grids
 * params: 
 * 	  offset        input, vertex displacement field, 3x(W+1)x(H+1)x(D+1) 
 *    points        input, all points, N_allx3
 *    distances     output, point to mesh distances for every grid for every topolopy, (WxHxD)xT 
 *    indices_all   output, to record which triangle in each topology is the nearest one for backpropagation, N_allxT
 *
 */
void point_topology_distance_cuda_forward( 
  at::Tensor offset,
  at::Tensor points,
  at::Tensor distances,
  at::Tensor indices_all) {
  CHECK_INPUT(offset);
  CHECK_INPUT(points);
  CHECK_INPUT(distances);
  CHECK_INPUT(indices_all);

  point_topology_distance_kernel_forward(offset, points, distances, indices_all);
}

/*
 * Backward function, calculating the gradients for the full offset map 
 * params: 
 *    grad_output   input, gradient on the output distances, (WxHxD)xT
 * 	  offset        input, vertex displacement field, 3x(W+1)x(H+1)x(D+1) 
 *    points        input, all points, N_allx3
 *    indices_all   input, recorded which triangle in each topology is the nearest one for backpropagation, N_allxT
 *    grad_offset   output, gradient on the full offset map, 3x(W+1)x(H+1)x(D+1)  
 *
 */	
void point_topology_distance_cuda_backward( 
  at::Tensor grad_output,
  at::Tensor offset,
  at::Tensor points,
  at::Tensor indices_all,
  at::Tensor grad_offset) {
  CHECK_INPUT(grad_offset);
  CHECK_INPUT(offset);
  CHECK_INPUT(points);
  CHECK_INPUT(indices_all);
  CHECK_INPUT(grad_offset);

  point_topology_distance_kernel_backward(grad_output, offset, points, indices_all, grad_offset);
}


// Python Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("curvature_constraint_forward", &curvature_constraint_cuda_forward, "curvature constraint foward (CUDA)");
  m.def("curvature_constraint_backward", &curvature_constraint_cuda_backward, "curvature constraint back (CUDA)");
  m.def("grid_pooling_forward", &grid_pooling_cuda_forward, "gridpooling foward (CUDA)");
  m.def("grid_pooling_backward", &grid_pooling_cuda_backward, "gridpooling back (CUDA)");
  m.def("occupancy_to_topology_forward", &occupancy_to_topology_cuda_forward, "occ2topo foward (CUDA)");
  m.def("occupancy_to_topology_backward", &occupancy_to_topology_cuda_backward, "occ2topo backward (CUDA)");
  m.def("occupancy_connectivity_forward", &occupancy_connectivity_cuda_forward, "occupancy connectivity foward (CUDA)");
  m.def("occupancy_connectivity_backward", &occupancy_connectivity_cuda_backward, "occupancy connectivity back (CUDA)");
  m.def("point_topology_distance_forward", &point_topology_distance_cuda_forward, "point topology distance foward (CUDA)");
  m.def("point_topology_distance_backward", &point_topology_distance_cuda_backward, "point topology distance back (CUDA)");
}
