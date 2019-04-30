#ifndef LIBRENDER_OFFSCREEN_H
#define LIBRENDER_OFFSCREEN_H

#include "GL/glew.h"
#include "GL/gl.h"
#include "GL/glu.h"
#include "GL/glut.h"

class OffscreenGL {

public:
  OffscreenGL(int maxHeight, int maxWidth);
  ~OffscreenGL();

private:
  static int glutWin;
  static bool glutInitialized;
  GLuint fb;
  GLuint renderTex;
  GLuint depthTex;
};


void renderDepthMesh(double *FM, int fNum, double *VM, int vNum, double *CM, double *intrinsics, int *imgSizeV, double *zNearFarV, unsigned char * imgBuffer, float *depthBuffer, bool *maskBuffer, double linewidth, bool coloring);

#endif