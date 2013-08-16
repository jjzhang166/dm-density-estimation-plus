#ifndef __CUDAKERNEL__
#define __CUDAKERNEL__
#include <cuda.h>
#include "../library/triangle.h"
#include "denrender.h"

#define RENDERTYPES_LIMIT 5

class Canvas{
public:

    
    //canvas data buffer
    float **deviceCanvasData;
    float **hostCanvasData;
    
    
    cudaError_t copyDeviceDataToHost(int zind);
    cudaError_t copyHostDataToDevice(int zind);
    
    
    int numRenderTypes;
    RenderType renderTypes[RENDERTYPES_LIMIT];
    
    
    int imagesize;
    float boxsize;
    
    float dx;   //dx = boxsize / imagesize
    
    //topleft and bottomright coordinates of the canvas
    Point2d topleft;
    Point2d bottomright;
    
    
};

__host__ cudaError_t drawTriangleOnGPU(Triangle triangle, float invVolum, Canvas canvas);
//render a triangle to the canvas
__global__ void renderTriangle(Triangle triangle, float invVolum, Canvas canvas, int xind, int yind);

#endif