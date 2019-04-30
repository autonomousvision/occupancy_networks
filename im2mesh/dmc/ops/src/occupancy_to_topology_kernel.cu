#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

__constant__ int T = 256;

__constant__ int acceptTopology[48] = {1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 17, 32, 34, 48, 51, 63, 64, 68, 96, 102, 111, 119, 127, 0, // upper 
	                        254, 253, 252, 251, 249, 247, 246, 243, 240, 239, 238, 223, 221, 207, 204, 192, 191, 187, 159, 153, 144, 136, 128, 255}; // bottom






// each row denotes a topology type
// each column denotes one of the vertex of a cell
// 2^8 = 256
__constant__ int occTable[256][8] = {{ 0,  0,  0,  0,  0,  0,  0,  0 },
       { 1,  0,  0,  0,  0,  0,  0,  0 },
       { 0,  1,  0,  0,  0,  0,  0,  0 },
       { 1,  1,  0,  0,  0,  0,  0,  0 },
       { 0,  0,  1,  0,  0,  0,  0,  0 },
       { 1,  0,  1,  0,  0,  0,  0,  0 },
       { 0,  1,  1,  0,  0,  0,  0,  0 },
       { 1,  1,  1,  0,  0,  0,  0,  0 },
       { 0,  0,  0,  1,  0,  0,  0,  0 },
       { 1,  0,  0,  1,  0,  0,  0,  0 },
       { 0,  1,  0,  1,  0,  0,  0,  0 },
       { 1,  1,  0,  1,  0,  0,  0,  0 },
       { 0,  0,  1,  1,  0,  0,  0,  0 },
       { 1,  0,  1,  1,  0,  0,  0,  0 },
       { 0,  1,  1,  1,  0,  0,  0,  0 },
       { 1,  1,  1,  1,  0,  0,  0,  0 },
       { 0,  0,  0,  0,  1,  0,  0,  0 },
       { 1,  0,  0,  0,  1,  0,  0,  0 },
       { 0,  1,  0,  0,  1,  0,  0,  0 },
       { 1,  1,  0,  0,  1,  0,  0,  0 },
       { 0,  0,  1,  0,  1,  0,  0,  0 },
       { 1,  0,  1,  0,  1,  0,  0,  0 },
       { 0,  1,  1,  0,  1,  0,  0,  0 },
       { 1,  1,  1,  0,  1,  0,  0,  0 },
       { 0,  0,  0,  1,  1,  0,  0,  0 },
       { 1,  0,  0,  1,  1,  0,  0,  0 },
       { 0,  1,  0,  1,  1,  0,  0,  0 },
       { 1,  1,  0,  1,  1,  0,  0,  0 },
       { 0,  0,  1,  1,  1,  0,  0,  0 },
       { 1,  0,  1,  1,  1,  0,  0,  0 },
       { 0,  1,  1,  1,  1,  0,  0,  0 },
       { 1,  1,  1,  1,  1,  0,  0,  0 },
       { 0,  0,  0,  0,  0,  1,  0,  0 },
       { 1,  0,  0,  0,  0,  1,  0,  0 },
       { 0,  1,  0,  0,  0,  1,  0,  0 },
       { 1,  1,  0,  0,  0,  1,  0,  0 },
       { 0,  0,  1,  0,  0,  1,  0,  0 },
       { 1,  0,  1,  0,  0,  1,  0,  0 },
       { 0,  1,  1,  0,  0,  1,  0,  0 },
       { 1,  1,  1,  0,  0,  1,  0,  0 },
       { 0,  0,  0,  1,  0,  1,  0,  0 },
       { 1,  0,  0,  1,  0,  1,  0,  0 },
       { 0,  1,  0,  1,  0,  1,  0,  0 },
       { 1,  1,  0,  1,  0,  1,  0,  0 },
       { 0,  0,  1,  1,  0,  1,  0,  0 },
       { 1,  0,  1,  1,  0,  1,  0,  0 },
       { 0,  1,  1,  1,  0,  1,  0,  0 },
       { 1,  1,  1,  1,  0,  1,  0,  0 },
       { 0,  0,  0,  0,  1,  1,  0,  0 },
       { 1,  0,  0,  0,  1,  1,  0,  0 },
       { 0,  1,  0,  0,  1,  1,  0,  0 },
       { 1,  1,  0,  0,  1,  1,  0,  0 },
       { 0,  0,  1,  0,  1,  1,  0,  0 },
       { 1,  0,  1,  0,  1,  1,  0,  0 },
       { 0,  1,  1,  0,  1,  1,  0,  0 },
       { 1,  1,  1,  0,  1,  1,  0,  0 },
       { 0,  0,  0,  1,  1,  1,  0,  0 },
       { 1,  0,  0,  1,  1,  1,  0,  0 },
       { 0,  1,  0,  1,  1,  1,  0,  0 },
       { 1,  1,  0,  1,  1,  1,  0,  0 },
       { 0,  0,  1,  1,  1,  1,  0,  0 },
       { 1,  0,  1,  1,  1,  1,  0,  0 },
       { 0,  1,  1,  1,  1,  1,  0,  0 },
       { 1,  1,  1,  1,  1,  1,  0,  0 },
       { 0,  0,  0,  0,  0,  0,  1,  0 },
       { 1,  0,  0,  0,  0,  0,  1,  0 },
       { 0,  1,  0,  0,  0,  0,  1,  0 },
       { 1,  1,  0,  0,  0,  0,  1,  0 },
       { 0,  0,  1,  0,  0,  0,  1,  0 },
       { 1,  0,  1,  0,  0,  0,  1,  0 },
       { 0,  1,  1,  0,  0,  0,  1,  0 },
       { 1,  1,  1,  0,  0,  0,  1,  0 },
       { 0,  0,  0,  1,  0,  0,  1,  0 },
       { 1,  0,  0,  1,  0,  0,  1,  0 },
       { 0,  1,  0,  1,  0,  0,  1,  0 },
       { 1,  1,  0,  1,  0,  0,  1,  0 },
       { 0,  0,  1,  1,  0,  0,  1,  0 },
       { 1,  0,  1,  1,  0,  0,  1,  0 },
       { 0,  1,  1,  1,  0,  0,  1,  0 },
       { 1,  1,  1,  1,  0,  0,  1,  0 },
       { 0,  0,  0,  0,  1,  0,  1,  0 },
       { 1,  0,  0,  0,  1,  0,  1,  0 },
       { 0,  1,  0,  0,  1,  0,  1,  0 },
       { 1,  1,  0,  0,  1,  0,  1,  0 },
       { 0,  0,  1,  0,  1,  0,  1,  0 },
       { 1,  0,  1,  0,  1,  0,  1,  0 },
       { 0,  1,  1,  0,  1,  0,  1,  0 },
       { 1,  1,  1,  0,  1,  0,  1,  0 },
       { 0,  0,  0,  1,  1,  0,  1,  0 },
       { 1,  0,  0,  1,  1,  0,  1,  0 },
       { 0,  1,  0,  1,  1,  0,  1,  0 },
       { 1,  1,  0,  1,  1,  0,  1,  0 },
       { 0,  0,  1,  1,  1,  0,  1,  0 },
       { 1,  0,  1,  1,  1,  0,  1,  0 },
       { 0,  1,  1,  1,  1,  0,  1,  0 },
       { 1,  1,  1,  1,  1,  0,  1,  0 },
       { 0,  0,  0,  0,  0,  1,  1,  0 },
       { 1,  0,  0,  0,  0,  1,  1,  0 },
       { 0,  1,  0,  0,  0,  1,  1,  0 },
       { 1,  1,  0,  0,  0,  1,  1,  0 },
       { 0,  0,  1,  0,  0,  1,  1,  0 },
       { 1,  0,  1,  0,  0,  1,  1,  0 },
       { 0,  1,  1,  0,  0,  1,  1,  0 },
       { 1,  1,  1,  0,  0,  1,  1,  0 },
       { 0,  0,  0,  1,  0,  1,  1,  0 },
       { 1,  0,  0,  1,  0,  1,  1,  0 },
       { 0,  1,  0,  1,  0,  1,  1,  0 },
       { 1,  1,  0,  1,  0,  1,  1,  0 },
       { 0,  0,  1,  1,  0,  1,  1,  0 },
       { 1,  0,  1,  1,  0,  1,  1,  0 },
       { 0,  1,  1,  1,  0,  1,  1,  0 },
       { 1,  1,  1,  1,  0,  1,  1,  0 },
       { 0,  0,  0,  0,  1,  1,  1,  0 },
       { 1,  0,  0,  0,  1,  1,  1,  0 },
       { 0,  1,  0,  0,  1,  1,  1,  0 },
       { 1,  1,  0,  0,  1,  1,  1,  0 },
       { 0,  0,  1,  0,  1,  1,  1,  0 },
       { 1,  0,  1,  0,  1,  1,  1,  0 },
       { 0,  1,  1,  0,  1,  1,  1,  0 },
       { 1,  1,  1,  0,  1,  1,  1,  0 },
       { 0,  0,  0,  1,  1,  1,  1,  0 },
       { 1,  0,  0,  1,  1,  1,  1,  0 },
       { 0,  1,  0,  1,  1,  1,  1,  0 },
       { 1,  1,  0,  1,  1,  1,  1,  0 },
       { 0,  0,  1,  1,  1,  1,  1,  0 },
       { 1,  0,  1,  1,  1,  1,  1,  0 },
       { 0,  1,  1,  1,  1,  1,  1,  0 },
       { 1,  1,  1,  1,  1,  1,  1,  0 },
       { 0,  0,  0,  0,  0,  0,  0,  1 },
       { 1,  0,  0,  0,  0,  0,  0,  1 },
       { 0,  1,  0,  0,  0,  0,  0,  1 },
       { 1,  1,  0,  0,  0,  0,  0,  1 },
       { 0,  0,  1,  0,  0,  0,  0,  1 },
       { 1,  0,  1,  0,  0,  0,  0,  1 },
       { 0,  1,  1,  0,  0,  0,  0,  1 },
       { 1,  1,  1,  0,  0,  0,  0,  1 },
       { 0,  0,  0,  1,  0,  0,  0,  1 },
       { 1,  0,  0,  1,  0,  0,  0,  1 },
       { 0,  1,  0,  1,  0,  0,  0,  1 },
       { 1,  1,  0,  1,  0,  0,  0,  1 },
       { 0,  0,  1,  1,  0,  0,  0,  1 },
       { 1,  0,  1,  1,  0,  0,  0,  1 },
       { 0,  1,  1,  1,  0,  0,  0,  1 },
       { 1,  1,  1,  1,  0,  0,  0,  1 },
       { 0,  0,  0,  0,  1,  0,  0,  1 },
       { 1,  0,  0,  0,  1,  0,  0,  1 },
       { 0,  1,  0,  0,  1,  0,  0,  1 },
       { 1,  1,  0,  0,  1,  0,  0,  1 },
       { 0,  0,  1,  0,  1,  0,  0,  1 },
       { 1,  0,  1,  0,  1,  0,  0,  1 },
       { 0,  1,  1,  0,  1,  0,  0,  1 },
       { 1,  1,  1,  0,  1,  0,  0,  1 },
       { 0,  0,  0,  1,  1,  0,  0,  1 },
       { 1,  0,  0,  1,  1,  0,  0,  1 },
       { 0,  1,  0,  1,  1,  0,  0,  1 },
       { 1,  1,  0,  1,  1,  0,  0,  1 },
       { 0,  0,  1,  1,  1,  0,  0,  1 },
       { 1,  0,  1,  1,  1,  0,  0,  1 },
       { 0,  1,  1,  1,  1,  0,  0,  1 },
       { 1,  1,  1,  1,  1,  0,  0,  1 },
       { 0,  0,  0,  0,  0,  1,  0,  1 },
       { 1,  0,  0,  0,  0,  1,  0,  1 },
       { 0,  1,  0,  0,  0,  1,  0,  1 },
       { 1,  1,  0,  0,  0,  1,  0,  1 },
       { 0,  0,  1,  0,  0,  1,  0,  1 },
       { 1,  0,  1,  0,  0,  1,  0,  1 },
       { 0,  1,  1,  0,  0,  1,  0,  1 },
       { 1,  1,  1,  0,  0,  1,  0,  1 },
       { 0,  0,  0,  1,  0,  1,  0,  1 },
       { 1,  0,  0,  1,  0,  1,  0,  1 },
       { 0,  1,  0,  1,  0,  1,  0,  1 },
       { 1,  1,  0,  1,  0,  1,  0,  1 },
       { 0,  0,  1,  1,  0,  1,  0,  1 },
       { 1,  0,  1,  1,  0,  1,  0,  1 },
       { 0,  1,  1,  1,  0,  1,  0,  1 },
       { 1,  1,  1,  1,  0,  1,  0,  1 },
       { 0,  0,  0,  0,  1,  1,  0,  1 },
       { 1,  0,  0,  0,  1,  1,  0,  1 },
       { 0,  1,  0,  0,  1,  1,  0,  1 },
       { 1,  1,  0,  0,  1,  1,  0,  1 },
       { 0,  0,  1,  0,  1,  1,  0,  1 },
       { 1,  0,  1,  0,  1,  1,  0,  1 },
       { 0,  1,  1,  0,  1,  1,  0,  1 },
       { 1,  1,  1,  0,  1,  1,  0,  1 },
       { 0,  0,  0,  1,  1,  1,  0,  1 },
       { 1,  0,  0,  1,  1,  1,  0,  1 },
       { 0,  1,  0,  1,  1,  1,  0,  1 },
       { 1,  1,  0,  1,  1,  1,  0,  1 },
       { 0,  0,  1,  1,  1,  1,  0,  1 },
       { 1,  0,  1,  1,  1,  1,  0,  1 },
       { 0,  1,  1,  1,  1,  1,  0,  1 },
       { 1,  1,  1,  1,  1,  1,  0,  1 },
       { 0,  0,  0,  0,  0,  0,  1,  1 },
       { 1,  0,  0,  0,  0,  0,  1,  1 },
       { 0,  1,  0,  0,  0,  0,  1,  1 },
       { 1,  1,  0,  0,  0,  0,  1,  1 },
       { 0,  0,  1,  0,  0,  0,  1,  1 },
       { 1,  0,  1,  0,  0,  0,  1,  1 },
       { 0,  1,  1,  0,  0,  0,  1,  1 },
       { 1,  1,  1,  0,  0,  0,  1,  1 },
       { 0,  0,  0,  1,  0,  0,  1,  1 },
       { 1,  0,  0,  1,  0,  0,  1,  1 },
       { 0,  1,  0,  1,  0,  0,  1,  1 },
       { 1,  1,  0,  1,  0,  0,  1,  1 },
       { 0,  0,  1,  1,  0,  0,  1,  1 },
       { 1,  0,  1,  1,  0,  0,  1,  1 },
       { 0,  1,  1,  1,  0,  0,  1,  1 },
       { 1,  1,  1,  1,  0,  0,  1,  1 },
       { 0,  0,  0,  0,  1,  0,  1,  1 },
       { 1,  0,  0,  0,  1,  0,  1,  1 },
       { 0,  1,  0,  0,  1,  0,  1,  1 },
       { 1,  1,  0,  0,  1,  0,  1,  1 },
       { 0,  0,  1,  0,  1,  0,  1,  1 },
       { 1,  0,  1,  0,  1,  0,  1,  1 },
       { 0,  1,  1,  0,  1,  0,  1,  1 },
       { 1,  1,  1,  0,  1,  0,  1,  1 },
       { 0,  0,  0,  1,  1,  0,  1,  1 },
       { 1,  0,  0,  1,  1,  0,  1,  1 },
       { 0,  1,  0,  1,  1,  0,  1,  1 },
       { 1,  1,  0,  1,  1,  0,  1,  1 },
       { 0,  0,  1,  1,  1,  0,  1,  1 },
       { 1,  0,  1,  1,  1,  0,  1,  1 },
       { 0,  1,  1,  1,  1,  0,  1,  1 },
       { 1,  1,  1,  1,  1,  0,  1,  1 },
       { 0,  0,  0,  0,  0,  1,  1,  1 },
       { 1,  0,  0,  0,  0,  1,  1,  1 },
       { 0,  1,  0,  0,  0,  1,  1,  1 },
       { 1,  1,  0,  0,  0,  1,  1,  1 },
       { 0,  0,  1,  0,  0,  1,  1,  1 },
       { 1,  0,  1,  0,  0,  1,  1,  1 },
       { 0,  1,  1,  0,  0,  1,  1,  1 },
       { 1,  1,  1,  0,  0,  1,  1,  1 },
       { 0,  0,  0,  1,  0,  1,  1,  1 },
       { 1,  0,  0,  1,  0,  1,  1,  1 },
       { 0,  1,  0,  1,  0,  1,  1,  1 },
       { 1,  1,  0,  1,  0,  1,  1,  1 },
       { 0,  0,  1,  1,  0,  1,  1,  1 },
       { 1,  0,  1,  1,  0,  1,  1,  1 },
       { 0,  1,  1,  1,  0,  1,  1,  1 },
       { 1,  1,  1,  1,  0,  1,  1,  1 },
       { 0,  0,  0,  0,  1,  1,  1,  1 },
       { 1,  0,  0,  0,  1,  1,  1,  1 },
       { 0,  1,  0,  0,  1,  1,  1,  1 },
       { 1,  1,  0,  0,  1,  1,  1,  1 },
       { 0,  0,  1,  0,  1,  1,  1,  1 },
       { 1,  0,  1,  0,  1,  1,  1,  1 },
       { 0,  1,  1,  0,  1,  1,  1,  1 },
       { 1,  1,  1,  0,  1,  1,  1,  1 },
       { 0,  0,  0,  1,  1,  1,  1,  1 },
       { 1,  0,  0,  1,  1,  1,  1,  1 },
       { 0,  1,  0,  1,  1,  1,  1,  1 },
       { 1,  1,  0,  1,  1,  1,  1,  1 },
       { 0,  0,  1,  1,  1,  1,  1,  1 },
       { 1,  0,  1,  1,  1,  1,  1,  1 },
       { 0,  1,  1,  1,  1,  1,  1,  1 },
       { 1,  1,  1,  1,  1,  1,  1,  1 }};


__constant__ int vertexTable[8][3]={ {0, 1, 0},
			       {1, 1, 0},
			       {1, 0, 0},
                               {0, 0, 0},
			       {0, 1, 1},
			       {1, 1, 1},
			       {1, 0, 1},
                               {0, 0, 1} };

namespace{
                               /**
 * convert the topology probabilites from the occupancy
 * parallel over every cell and every topology
 */
 template <typename scalar_t>
 __global__ void occupancy_to_topology_kernel(const scalar_t *occupancy, scalar_t *topology){
  // int W = gridDim.x;
  int H = gridDim.y;
  int D = gridDim.z;

  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;

  int t = threadIdx.x;
  // return probabilities of all 256 topologies
  int topology_ind = t; 

  float p_occ[2][8];
  for (int v=0; v<8; v++){
    p_occ[0][v] = occupancy[ (i+vertexTable[v][0])*(H+1)*(D+1) + (j+vertexTable[v][1])*(D+1) + k+vertexTable[v][2] ]; 
    p_occ[1][v] = 1-p_occ[0][v]; 
  }


  float p_accumu = 1.0;
  for (int v=0; v<8; v++){
      p_accumu = p_accumu*p_occ[occTable[topology_ind][v]][v]; 
  }
  topology[ (i*H*D+j*D+k)*T + t ] = p_accumu;
}


/**
 * propagate the gradient from the topology probabilities to occupancy status
 * parallel over every cell and every topology
 */
 template <typename scalar_t>
 __global__ void grad_occupancy_to_topology_kernel(const scalar_t *grad_output, const scalar_t *occupancy, scalar_t *topology, scalar_t *grad_occupancy){
  // int W = gridDim.x;
  int H = gridDim.y;
  int D = gridDim.z;

  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;

  int t = threadIdx.x;
  // return probabilities of all 256 topologies
  int topology_ind = t; 

  scalar_t p_occ[2][8];
  for (int v=0; v<8; v++){
    p_occ[0][v] = occupancy[ (i+vertexTable[v][0])*(H+1)*(D+1) + (j+vertexTable[v][1])*(D+1) + k+vertexTable[v][2] ]; 
    p_occ[1][v] = 1-p_occ[0][v]; 
  }


  //float p_accumu = topology[ (i*H+j)*T + t ];
  scalar_t grad_accumu = grad_output[ (i*H*D+j*D+k)*T + t ];
  // propagate the gradient to four occupancy corners
  scalar_t sign;
  for (int v=0; v<8; v++){
    if (occTable[topology_ind][v]==0){
            sign=1.0;
    }else{
            sign=-1.0;
    } 
  
    // re-calculate the probability excluding the current vertex
    // didn't use p_accumu/p_occ[occTable[t][v]][v] for numerial stability
    // TODO: find a better solution
    scalar_t p_accumu = 1.0;
    for (int v_=0; v_<8; v_++){
	if (v_==v) continue;
        p_accumu = p_accumu*p_occ[occTable[topology_ind][v_]][v_]; 
    }
    atomicAdd(&grad_occupancy[ (i+vertexTable[v][0])*(H+1)*(D+1) + (j+vertexTable[v][1])*(D+1) + k+vertexTable[v][2] ], sign*grad_accumu*p_accumu);
  }

}

} //namespace

/*
 * Forward function, compute the topology probability given the occupancy probability 
 * params: 
 *     occupancy   input, (W+1)x(H+1)
 *     topology    output, probability of all topologies types we care about (WxH)xT
 *
 */	
void occupancy_to_topology_kernel_forward( 
    at::Tensor occupancy, 
    at::Tensor topology ){

  int W = occupancy.size(0) - 1;
  int H = occupancy.size(1) - 1;
  int D = occupancy.size(2) - 1;
  int T = topology.size(1);

  dim3 dimGrid(W, H, D);
  dim3 dimBlock(T, 1, 1);
  // lauch the kernel
  AT_DISPATCH_FLOATING_TYPES(topology.type(), "occ2topo_foward", ([&] {
    occupancy_to_topology_kernel<scalar_t><<< dimGrid, dimBlock>>>(
        occupancy.data<scalar_t>(),
        topology.data<scalar_t>());
  }));
}

/*
 * Backward function, backpropagate the gradient from topology to occupancy 
 * params: 
 *     grad_output     input, gradient on the topology probability, (WxH)xT
 *  	 occupancy 	     input, (W+1)x(H+1)
 *     topology     	 input, probability of all topologies types we care about (WxH)xT
 *     grad_occupancy  output, gradient on the occupancy map, (W+1)x(H+1) 
 *
 */	
void occupancy_to_topology_kernel_backward( 
    at::Tensor grad_output,
    at::Tensor occupancy, 
    at::Tensor topology,
    at::Tensor grad_occupancy){

  int W = occupancy.size(0) - 1;
  int H = occupancy.size(1) - 1;
  int D = occupancy.size(2) - 1;
  int T = topology.size(1);

  dim3 dimGrid(W, H, D);
  dim3 dimBlock(T, 1, 1);
  // lauch the kernel

  grad_occupancy_to_topology_kernel<float><<< dimGrid, dimBlock>>>(
      grad_output.data<float>(),
      occupancy.data<float>(),
      topology.data<float>(),
      grad_occupancy.data<float>());
}
