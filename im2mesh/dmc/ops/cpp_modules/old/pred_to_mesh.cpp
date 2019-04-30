#include <torch/torch.h>
#include <TH/TH.h>
#include <stdio.h>
#include "commons.h"


// considered all topologies with 4 triangles during visualization
static int visTopology[2][140]={{0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 23, 25, 27, 29, 31, 32, 34, 35, 38, 39, 43, 46, 47, 48, 49, 50, 51, 54, 55, 57, 59, 63, 64, 68, 70, 71, 76, 77, 78, 79, 95, 96, 98, 99, 100, 102, 103, 108, 110, 111, 112, 113, 114, 115, 116, 118, 119, 123, 126, 127, 128, 136, 137, 139, 140, 141, 142, 143, 144, 145, 147, 152, 153, 155, 156, 157, 159, 175, 176, 177, 178, 179, 183, 184, 185, 187, 189, 191, 192, 196, 198, 200, 201, 204, 205, 206, 207, 208, 209, 212, 216, 217, 219, 220, 221, 222, 223, 224, 226, 228, 230, 231, 232, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255},
{0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 2, 3, 3, 2, 1, 2, 3, 4, 3, 4, 4, 3, 1, 2, 3, 3, 4, 4, 4, 3, 2, 3, 3, 2, 4, 3, 4, 3, 2, 1, 2, 3, 4, 3, 4, 4, 3, 4, 2, 3, 4, 3, 2, 3, 4, 3, 2, 3, 4, 4, 3, 4, 3, 2, 4, 4, 1, 1, 2, 3, 4, 3, 4, 4, 3, 2, 3, 4, 3, 2, 3, 4, 3, 2, 4, 3, 4, 4, 3, 4, 4, 3, 2, 4, 1, 2, 3, 4, 3, 4, 2, 3, 3, 2, 3, 4, 4, 4, 3, 4, 3, 2, 4, 1, 3, 4, 4, 3, 4, 4, 3, 4, 2, 1, 2, 3, 3, 2, 3, 4, 2, 1, 3, 2, 4, 1, 2, 1, 1, 0}};

/**
 * convert the topology probability and vertex displacement field to a mesh by
 * selecting the topology with maximum probability in every cell
 * params:
 * 	offset 		vertex displacement field
 * 	topology 	topology probabilities
 * 	vertices_all	vertices locations for all triangles in topologies with maximum probabilities
 * 			note there might be duplications and the unique vertices will be extracted afterwards
 * 	faces_all 	faces represented by the indices in vertices_all
 * 	vertice_number	record the number of vertices as we initialzed the vertices_all with a fixed length 
 * 	face_number 	record the number of faces as we initialized the faces_all with a fixed length 
 */
int pred_to_mesh(THFloatTensor offset, THLongTensor *topology, THFloatTensor *vertices_all, THFloatTensor *faces_all, THLongTensor *vertice_number, THLongTensor *face_number){
  // data format check
  if (THFloatTensor_nDimension(offset)!=4 ||  THLongTensor_nDimension(topology)!=3 ){
    printf("Invalid nDimension!\n");
    printf("Expected 4, 3, received %d, %d \n", THFloatTensor_nDimension(offset), THLongTensor_nDimension(topology));
    return 0;
  }
  int W,H,D;
  W = THFloatTensor_size(offset,1)-1; 
  H = THFloatTensor_size(offset,2)-1; 
  D = THFloatTensor_size(offset,3)-1; 

  int vertice_cnt=0;
  int face_cnt=0;

  for (int i=0; i<W; i++){
    for (int j=0; j<H; j++){
      for (int k=0; k<D; k++){

        int t = THLongTensor_get3d(topology, i, j, k);
        float offset_c[3]={(float)i, (float)j, (float)k};
        
        THFloatTensor *vertices = offset_to_vertices(offset, i, j, k);
        for (int tri_ind = 0; tri_ind<visTopology[1][t]; tri_ind++){
            for (int vertex_ind = 0; vertex_ind<3; vertex_ind++){
        	
        	// copy the vertex coordinates to the triangle matrix
        	int topology_ind = visTopology[0][t];
                 
        	for (int _i=0; _i<3; _i++){
	            THFloatTensor_set2d( vertices_all, vertice_cnt, _i, THFloatTensor_get2d(vertices, _i, triTable[topology_ind][tri_ind*3+vertex_ind])+offset_c[_i] );
                }

              THFloatTensor_set2d( faces_all, face_cnt, vertex_ind, vertice_cnt);
              vertice_cnt++;
            }
            face_cnt++;

        }
        THFloatTensor_free(vertices);
        
      }
    }
  }

  THLongTensor_set1d(vertice_number, 0, vertice_cnt);
  THLongTensor_set1d(face_number, 0, face_cnt);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pred2mesh", &pred_to_mesh, "Lpred2mesh");
}