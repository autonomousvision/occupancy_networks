#include "offscreen.h"
#include <cstdio>

int OffscreenGL::glutWin = -1;
bool OffscreenGL::glutInitialized = false;

OffscreenGL::OffscreenGL(int maxHeight, int maxWidth) {

  if (!glutInitialized) {
    int argc = 1;
    char *argv = "test";
    glutInit(&argc, &argv);
    glutInitialized = true;
  }

  glutInitDisplayMode(GLUT_DEPTH | GLUT_SINGLE | GLUT_RGBA);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(maxWidth, maxHeight);

  // create or set window & off-screen framebuffer
  if (glutWin < 0) {

    glutWin = glutCreateWindow("OpenGL");
    glutHideWindow();
    glewInit();
    glGenFramebuffersEXT(1, &fb);

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb);
    glGenTextures(1, &renderTex);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, renderTex);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGB, maxWidth, maxHeight,
            0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    glGenTextures(1, &depthTex);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, depthTex);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_DEPTH24_STENCIL8, maxWidth, maxHeight, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);

    glGenFramebuffersEXT(1, &fb);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_RECTANGLE_ARB, renderTex, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_RECTANGLE_ARB, depthTex, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT|GL_DEPTH_ATTACHMENT_EXT);
  } else {
    glutSetWindow(glutWin);
  }
}

OffscreenGL::~OffscreenGL() {
}


GLuint createDisplayList(double *fM, int fNum, double *vM, int vNum, double *cM, unsigned int colorModFactor, double linewidth, bool coloring) {

  GLuint theShape;
  int i;
  unsigned int channelCapacity, channelCapacity2;
  double *fp;
  int vIndex, fNum2;
  fNum2 = fNum*2;

  channelCapacity = 256 / colorModFactor;
  channelCapacity2 = channelCapacity * channelCapacity;

  theShape = glGenLists(1);

  glNewList(theShape, GL_COMPILE);

  if (linewidth>0.1) {
    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
    glLineWidth(linewidth);
  }

  glBegin(GL_TRIANGLES);
  for (i = 1; i <= fNum; i++) {
    fp = fM + i-1;

    vIndex = (int)fp[0] - 1;
    if (coloring)  glColor3ub(cM[vIndex], cM[vIndex + vNum], cM[vIndex + 2*vNum]);
    glVertex3d(vM[vIndex], vM[vIndex + vNum], vM[vIndex + 2*vNum]);

    vIndex = (int)fp[fNum] - 1;
    if (coloring)  glColor3ub(cM[vIndex], cM[vIndex + vNum], cM[vIndex + 2*vNum]);
    glVertex3d(vM[vIndex], vM[vIndex + vNum], vM[vIndex + 2*vNum]);

    vIndex = (int)fp[fNum2] - 1;
    if (coloring)  glColor3ub(cM[vIndex], cM[vIndex + vNum], cM[vIndex + 2*vNum]);
    glVertex3d(vM[vIndex], vM[vIndex + vNum], vM[vIndex + 2*vNum]);
  }
  glEnd();
  if (linewidth>0.1)
    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
  glEndList();

  return theShape;
}

void cameraSetup(double zNear, double zFar, double *intrinsics, unsigned int imgHeight, unsigned int imgWidth) {

  double viewMat[] = {1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1};
  double fcv[] = {intrinsics[0], intrinsics[1]};
  double ccv[] = {intrinsics[2], intrinsics[3]};

  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_TEXTURE_2D);

  glMatrixMode(GL_MODELVIEW);
  glLoadMatrixd(viewMat);

  double left = - ccv[0] / fcv[0] * zNear;
  double bottom = (ccv[1] - (double)(imgHeight-1)) / fcv[1] * zNear;
  double right = ((double)imgWidth - 1.0 - ccv[0]) / fcv[0] * zNear;
  double top = ccv[1] / fcv[1] * zNear;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glFrustum(left, right, bottom, top, zNear, zFar);
  glViewport(0, 0, imgWidth, imgHeight);
}

void drawPatchToDepthBuffer(GLuint listName, unsigned char *imageBuffer, float *depthBuffer, bool *maskBuffer,
        unsigned int imgHeight, unsigned int imgWidth, double *zNearFarV, bool coloring = true) {

  glCallList(listName);
  glFlush();

  // bug fix for Nvidia
  unsigned int paddedWidth = imgWidth % 4;
  if (paddedWidth != 0) paddedWidth = 4 - paddedWidth + imgWidth;
  else                  paddedWidth = imgWidth;

  // Read off of the depth buffer
  float *dataBuffer_depth = (float *)malloc(paddedWidth * imgHeight * sizeof(GL_FLOAT));
  glReadPixels(0, 0, paddedWidth, imgHeight, GL_DEPTH_COMPONENT, GL_FLOAT, dataBuffer_depth);

  // Read off of the color buffer
  GLubyte *dataBuffer_rgb = (GLubyte *)malloc(3* paddedWidth * imgHeight * sizeof(GLubyte));
  if (coloring)
    glReadPixels(0, 0, paddedWidth, imgHeight, GL_RGB, GL_UNSIGNED_BYTE, dataBuffer_rgb);

  // reorder the pixel data for the opengl to matlab conversion
  unsigned int matlabImgIndex = 0;
  unsigned int oglImageIndex = 0;

  float n = zNearFarV[0];
  float f = zNearFarV[1];
  for (int j = 0; j < imgWidth; j++) {
    for (int i = 0; i < imgHeight; i++, matlabImgIndex++) {
      oglImageIndex = (j + (imgHeight-1-i) * paddedWidth);
      float depth = dataBuffer_depth[oglImageIndex];

      // render mask: indicating points inside the clipped plane
      maskBuffer[matlabImgIndex] = depth<1;

      // render depth
      depthBuffer[matlabImgIndex] = -f*n/(depth*(f-n)-f);

      // render color
      if (coloring) {
        imageBuffer[matlabImgIndex] = (unsigned char) dataBuffer_rgb[oglImageIndex*3];
        imageBuffer[matlabImgIndex+imgWidth*imgHeight] = (unsigned char) dataBuffer_rgb[oglImageIndex*3+1];
        imageBuffer[matlabImgIndex+imgWidth*imgHeight*2] = (unsigned char) dataBuffer_rgb[oglImageIndex*3+2];
      }
    }
  }

  free(dataBuffer_depth);
  free(dataBuffer_rgb);
}

void renderDepthMesh(double *FM, int fNum, double *VM, int vNum, double *CM, double *intrinsics, int *imgSizeV, double *zNearFarV, unsigned char * imgBuffer, float *depthBuffer, bool *maskBuffer, double linewidth, bool coloring) {
  //createGLContext();
  OffscreenGL offscreenGL(imgSizeV[0], imgSizeV[1]);
  cameraSetup(zNearFarV[0], zNearFarV[1], intrinsics, imgSizeV[0], imgSizeV[1]);
  GLuint list = createDisplayList(FM, fNum, VM, vNum, CM, 1, linewidth, coloring);
  drawPatchToDepthBuffer(list, imgBuffer, depthBuffer, maskBuffer, imgSizeV[0], imgSizeV[1], zNearFarV, coloring);
  if (list) {
    glDeleteLists(list, 1);
    list = 0;
  }
  //deleteGLContext();
}
