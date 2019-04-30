#include <TH/TH.h>
#include "commons.h"

/**
 * convert vertex displacement field to vertices locations
 * params:
 * 	offset	vertex displacement field, 3xWxHxD
 * 	x 	indice of a cell in the full grid	
 * 	y 	indice of a cell in the full grid	
 * 	z 	indice of a cell in the full grid	
 * return:
 * 	vertices 	the location of 12 vertices for the specific cell, 3x12 
 *
 */
THFloatTensor* offset_to_vertices(THFloatTensor *offset, int x, int y, int z){
  THFloatTensor *vertices = THFloatTensor_newWithSize2d(3, 12); 

  // #0
  THFloatTensor_set2d(vertices, 0, 0, 0.5-THFloatTensor_get4d(offset, 0, x+1, y+1, z  ));
  THFloatTensor_set2d(vertices, 1, 0, 1.0);
  THFloatTensor_set2d(vertices, 2, 0, 0.0);
  // #1
  THFloatTensor_set2d(vertices, 0, 1, 1.0);
  THFloatTensor_set2d(vertices, 1, 1, 0.5-THFloatTensor_get4d(offset, 1, x+1, y+1, z  ));
  THFloatTensor_set2d(vertices, 2, 1, 0.0);
  // #2
  THFloatTensor_set2d(vertices, 0, 2, 0.5-THFloatTensor_get4d(offset, 0, x+1, y  , z  ));
  THFloatTensor_set2d(vertices, 1, 2, 0.0);
  THFloatTensor_set2d(vertices, 2, 2, 0.0);
  // #3
  THFloatTensor_set2d(vertices, 0, 3, 0.0);
  THFloatTensor_set2d(vertices, 1, 3, 0.5-THFloatTensor_get4d(offset, 1, x  , y+1, z  ));
  THFloatTensor_set2d(vertices, 2, 3, 0.0);

  // #4
  THFloatTensor_set2d(vertices, 0, 4, 0.5-THFloatTensor_get4d(offset, 0, x+1, y+1, z+1));
  THFloatTensor_set2d(vertices, 1, 4, 1.0);
  THFloatTensor_set2d(vertices, 2, 4, 1.0);
  // #5
  THFloatTensor_set2d(vertices, 0, 5, 1.0);
  THFloatTensor_set2d(vertices, 1, 5, 0.5-THFloatTensor_get4d(offset, 1, x+1, y+1, z+1));
  THFloatTensor_set2d(vertices, 2, 5, 1.0);
  // #6
  THFloatTensor_set2d(vertices, 0, 6, 0.5-THFloatTensor_get4d(offset, 0, x+1, y  , z+1));
  THFloatTensor_set2d(vertices, 1, 6, 0.0);
  THFloatTensor_set2d(vertices, 2, 6, 1.0);
  // #7
  THFloatTensor_set2d(vertices, 0, 7, 0.0);
  THFloatTensor_set2d(vertices, 1, 7, 0.5-THFloatTensor_get4d(offset, 1, x  , y+1, z+1));
  THFloatTensor_set2d(vertices, 2, 7, 1.0);

  // #8
  THFloatTensor_set2d(vertices, 0, 8, 0.0);
  THFloatTensor_set2d(vertices, 1, 8, 1.0);
  THFloatTensor_set2d(vertices, 2, 8, 0.5-THFloatTensor_get4d(offset, 2, x  , y+1, z+1));
  // #9
  THFloatTensor_set2d(vertices, 0, 9, 1.0);
  THFloatTensor_set2d(vertices, 1, 9, 1.0);
  THFloatTensor_set2d(vertices, 2, 9, 0.5-THFloatTensor_get4d(offset, 2, x+1, y+1, z+1));
  // #10
  THFloatTensor_set2d(vertices, 0, 10, 1.0);
  THFloatTensor_set2d(vertices, 1, 10, 0.0);
  THFloatTensor_set2d(vertices, 2, 10, 0.5-THFloatTensor_get4d(offset, 2, x+1, y  , z+1));
  // #11
  THFloatTensor_set2d(vertices, 0, 11, 0.0);
  THFloatTensor_set2d(vertices, 1, 11, 0.0);
  THFloatTensor_set2d(vertices, 2, 11, 0.5-THFloatTensor_get4d(offset, 2, x  , y  , z+1));
  return vertices;
}

/**
 * get points in a specific cell 
 * params:
 * 	points	all points in the grid, Nx3	
 * 	i 	the offset of the specific cell
 * 	j 	the offset of the specific cell
 * 	k 	the offset of the specific cell
 * return:
 * 	indices		a binary 1D tensor indicating if a point is in a specific cell or not, N 
 *
 */
THLongTensor* points_in_grid(THFloatTensor *points, float i, float j, float k){
  int N=THFloatTensor_size(points, 0);
  THLongTensor *indices = THLongTensor_new();

  THByteTensor *mask = THByteTensor_newWithSize1d(N);
  THByteTensor_zero(mask);
  for (int p=0; p<N; p++){
     if (THFloatTensor_get2d(points, p, 0) >= i && THFloatTensor_get2d(points, p, 0) < i+1 &&
         THFloatTensor_get2d(points, p, 1) >= j && THFloatTensor_get2d(points, p, 1) < j+1 && 
         THFloatTensor_get2d(points, p, 2) >= k && THFloatTensor_get2d(points, p, 2) < k+1)
	    THByteTensor_set1d(mask, p, 1); 
  }

  THByteTensor_nonzero(indices, mask);

  THLongTensor_squeeze(indices, indices);
  THByteTensor_free(mask);
  return indices;
}
