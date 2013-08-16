#include <cstdio>
#include <cstdlib>

#include <cmath>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#if defined(_WIN32) || defined(_WIN64)
//#include "gettimeofday_win.h"
#else
#include <unistd.h>
#include <sys/time.h>
#endif


#include "types.h"

#include "tetracut.h"

#include "denrender.h"
#include "cudakernel.h"

using namespace std;

int num_of_rendertype = 0;

Canvas canvas;

//the depth of the triangle buffer
const int DenRender::VERTEXBUFFERDEPTH = 64 * 1024;

//at most render 4 different render type at the same time
const int DenRender::NUM_OF_RENDERTRYPE_LIMIT = 5;



Triangle * vertexbuffer_;
float * volumebuffer_;
int * vertexIds_;

DenRender::DenRender(int imagesize,
                     float boxsize,
                     float startz,
                     float dz,
                     int numplane,
                     vector<RenderType> rentypes){
    
    
    cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        exit(1);
    }
    
    imagesize_ = imagesize;
    boxsize_ = boxsize;
    numplanes_ = numplane;
    dz_ = dz;
    startz_ = startz;
    rentypes_ = rentypes;
    
    
    num_of_rendertype = rentypes.size();
    result_ = new float[imagesize * imagesize * numplanes_ * num_of_rendertype];
    
    vertexbuffer_ = new Triangle[VERTEXBUFFERDEPTH * numplanes_];
    volumebuffer_ = new float[VERTEXBUFFERDEPTH * numplanes_];
    vertexIds_ = new int[numplanes_];
    
    
    for(int i = 0; i < imagesize * imagesize * numplanes_ * num_of_rendertype; i++){
        result_[i] = 0.0f;
    }
    
    canvas.imagesize = imagesize;
    canvas.boxsize = boxsize;
    canvas.dx = boxsize / imagesize;
    canvas.topleft.x = 0;
    canvas.topleft.y = 0;
    canvas.bottomright.x = boxsize;
    canvas.bottomright.y = boxsize;
    canvas.numRenderTypes = num_of_rendertype;
    //canvas.renderTypes = rentypes;
    if(rentypes.size() >= 1)
        canvas.type1 = rentypes[0];
    if(rentypes.size() >= 2)
        canvas.type2 = rentypes[1];
    if(rentypes.size() >= 3)
        canvas.type3 = rentypes[2];
    if(rentypes.size() >= 4)
        canvas.type4 = rentypes[3];
    if(rentypes.size() >= 5)
        canvas.type5 = rentypes[4];

    canvas.hostCanvasData = new float*[num_of_rendertype];
    canvas.deviceCanvasData = new float*[num_of_rendertype];
    
    for(int i = 0; i < num_of_rendertype; i++){
        canvas.hostCanvasData[i] = result_ +
                            imagesize * imagesize * numplanes_ * i;
        cudaStatus = cudaMalloc((void**)&canvas.deviceCanvasData[i],
                                imagesize * imagesize * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed");
            exit(1);
        }
    }
    
    
}

DenRender::~DenRender(){
    delete result_;
    
    for(int i = 0; i < num_of_rendertype; i++){
        cudaFree(canvas.deviceCanvasData[i]);
    }
    
    delete canvas.deviceCanvasData;
    delete canvas.hostCanvasData;
    
    delete vertexbuffer_;
    delete volumebuffer_;
    delete vertexIds_;
}

void DenRender::rend(Tetrahedron & tetra){
    cutter.setTetrahedron(&tetra);
    
    if(startz_ > tetra.v4.z || tetra.v1.z > startz_ + numplanes_ * dz_){
        return;
    }
    
    int starti = max(floor((tetra.v1.z  - startz_ )/ dz_), 0.0f);
    int endi = min(ceil((tetra.v4.z  - startz_) / dz_), (float)numplanes_);
    
    for(int i = starti; i < endi; i++){
        float z = startz_ + dz_ * i;
        
        int tris = cutter.cut(z);
        for(int j = 0; j < tris; j++){
            //density, stream number, velocity_x, velocity_y, velocity_z
            vertexbuffer_[i * VERTEXBUFFERDEPTH + vertexIds_[i]].a
                = cutter.getTriangle(j).a;
            vertexbuffer_[i * VERTEXBUFFERDEPTH + vertexIds_[i]].b
                = cutter.getTriangle(j).b;
            vertexbuffer_[i * VERTEXBUFFERDEPTH + vertexIds_[i]].c
                = cutter.getTriangle(j).c;
            vertexbuffer_[i * VERTEXBUFFERDEPTH + vertexIds_[i]].val1
                = cutter.getTriangle(j).val1;
            vertexbuffer_[i * VERTEXBUFFERDEPTH + vertexIds_[i]].val2
                = cutter.getTriangle(j).val2;
            vertexbuffer_[i * VERTEXBUFFERDEPTH + vertexIds_[i]].val3
                = cutter.getTriangle(j).val3;
            volumebuffer_[i * VERTEXBUFFERDEPTH + vertexIds_[i]]
                = tetra.invVolume;
            
            vertexIds_[i] ++;
            
            if(vertexIds_[i] >= VERTEXBUFFERDEPTH){
                rendplane(i);
            }
        }
    }

}

void DenRender::finish(){
    for(int i = 0; i < numplanes_; i++){
        //printf("%d \n", vertexIds_[i]);
        if(vertexIds_[i] > 0){
            rendplane(i);
        }
    }
}

float * DenRender::getResult(){
    return result_;
}


void DenRender::rendplane(int i){
    for(int j = 0; j < vertexIds_[i]; j++){
        drawTriangleOnGPU(vertexbuffer_[VERTEXBUFFERDEPTH * i + j],
                          volumebuffer_[VERTEXBUFFERDEPTH * i + j],
                          canvas);
    }
    vertexIds_[i] = 0;
}

void DenRender::init(){
    
}
