#ifndef __CUDAKERNEL__
#define __CUDAKERNEL__
#include <cuda.h>
#include "triangle.h"
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
    RenderType type1;
    RenderType type2;
    RenderType type3;
    RenderType type4;
    RenderType type5;

    CUDA_CALLABLE_MEMBER RenderType renderTypes(int i){
        switch(i){
            case 0:
                return type1;
            case 1:
                return type2;
            case 2:
                return type3;
            case 3:
                return type4;
            case 4:
                return type5;
        }
        return type1;
    };
    
    
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
