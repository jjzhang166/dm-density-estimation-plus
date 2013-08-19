#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudakernel.h"

#include "triangle.cpp"
#include "tetrahedron.cpp"

//__device__ static const float divider[] = {1.0, 0.5, 0.25};

__host__ cudaError_t drawTriangleOnGPU(Triangle triangle, float invVolum, Canvas canvas){
    Point2d topleft;
    Point2d bottomright;
    
    topleft.x = min(min(triangle.a.x, triangle.b.x), triangle.c.x);
    topleft.y = min(min(triangle.a.y, triangle.b.y), triangle.c.y);
    bottomright.x = max(max(triangle.a.x, triangle.b.x), triangle.c.x);
    bottomright.y = max(max(triangle.a.y, triangle.b.y), triangle.c.y);
    
    Point2d vt = topleft - canvas.topleft;
    Point2d vb = bottomright - canvas.topleft;
    
    int xind = (int) floor(vt.x / canvas.dx);
    int yind = (int) floor(vt.y / canvas.dx);
    int xend = (int) ceil(vb.x / canvas.dx);
    int yend = (int) ceil(vb.y / canvas.dx);
    
    dim3 block_size;
    block_size.x = 4;
    block_size.y = 4;
    
    dim3 grid_size;
    grid_size.x = (int)ceil((float)(xend - xind) / (float)block_size.x);
    grid_size.y = (int)ceil((float)(yend - yind) / (float)block_size.y);
    
    renderTriangle<<<grid_size, block_size>>>(triangle, invVolum, canvas, xind, yind);
    
    return cudaThreadSynchronize();
    
}


cudaError_t Canvas::copyDeviceDataToHost(int zind){
    cudaError_t err = cudaSuccess;
    for(int i = 0; i < numRenderTypes; i++){
        cudaError_t err1 = cudaMemcpy(hostCanvasData[i]
                                      + zind * imagesize * imagesize,
                                      deviceCanvasData[i],
                                      imagesize * imagesize * sizeof(float), cudaMemcpyDeviceToHost);
        if(err1 != cudaSuccess){
            err = err1;
        }
    }
    
    return err;
}


cudaError_t Canvas::copyHostDataToDevice(int zind){
    cudaError_t err = cudaSuccess;
    for(int i = 0; i < numRenderTypes; i++){
        cudaError_t err1 = cudaMemcpy(deviceCanvasData[i],
                                      hostCanvasData[i]
                                       + zind * imagesize * imagesize,
                                      imagesize * imagesize * sizeof(float),cudaMemcpyHostToDevice);
        if(err1 != cudaSuccess){
            err = err1;
        }
    }
    
    return err;
}

//return -1 no
//return 0 yes
//return 1 on the edge
//return 2 on the vertex
__device__ int isInTriangle(Triangle & triangle, Point2d &p, float &u, float &v){
    if(p.x > triangle.a.x && p.x > triangle.b.x && p.x > triangle.c.x)
        return -1;
    
    if(p.x < triangle.a.x && p.x < triangle.b.x && p.x < triangle.c.x)
        return -1;
    
    if(p.y > triangle.a.y && p.y > triangle.b.y && p.y > triangle.c.y)
        return -1;
    
    if(p.y < triangle.a.y && p.y < triangle.b.y && p.y < triangle.c.y)
        return -1;
    
    if(p.x == triangle.a.x && p.y == triangle.a.y)
        return 2;
    
    if(p.x == triangle.b.x && p.y == triangle.b.y)
        return 2;
    
    if(p.x == triangle.c.x && p.y == triangle.c.y)
        return 2;
    
    // Compute vectors
    Point2d v0 = triangle.c - triangle.a;
    Point2d v1 = triangle.b - triangle.a;
    Point2d v2 = p - triangle.a;
    
    // Compute dot products
    float dot00 = v0.dot(v0);
    float dot01 = v0.dot(v1);
    float dot02 = v0.dot(v2);
    float dot11 = v1.dot(v1);
    float dot12 = v1.dot(v2);
    
    // Compute barycentric coordinates
    float volume = (dot00 * dot11 - dot01 * dot01);
    if(volume == 0)
        return 2;            //excluded by the vetex test
    //float invDenom = 1.0 / volume;
    u = (dot11 * dot02 - dot01 * dot12) / volume;
    v = (dot00 * dot12 - dot01 * dot02) / volume;
    
    // Check if point is in triangle
    if( (u > 0) && (v > 0) && (u + v < 1)){
        return 0;
    }else if((u == 0) || (v == 0) || (u + v) == 1){
        return 1;
    }else{
        return -1;
    }
    
}

__global__ void renderTriangle(Triangle triangle, float invVolum, Canvas canvas, int xind, int yind){
    const float divider[] = {1.0, 0.5, 0.25};
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    Point2d p;
    p.x = (idx + xind) * canvas.dx + canvas.topleft.x;
    p.y = (idy + yind) * canvas.dx + canvas.topleft.y;
    
    float u, v;
    int inTriangle = isInTriangle(triangle, p, u, v);
    if(inTriangle != -1){
        int ind = idx + xind + (idy + yind) * canvas.imagesize;
        Point velocity = triangle.val1 * (1.0 - u - v)
            + triangle.val2 * u + triangle.val3 * v;
        float values[] = {invVolum, 1.0, velocity.x, velocity.y, velocity.z};
        
        for(int i = 0; i < canvas.numRenderTypes; i++){
            canvas.deviceCanvasData[i][ind] += values[canvas.renderTypes(i)] *
                                            divider[inTriangle];
        }
    }

}



