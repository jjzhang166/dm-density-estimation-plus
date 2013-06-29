#ifdef __APPLE__
//#include <glew.h>
#include <GLUT/glut.h> // darwin uses glut.h rather than GL/glut.h
#else
#include <GL/glew.h>
#include <GL/glut.h>
#endif

#include <cmath>

#include <unistd.h>
#include <sys/time.h>

#include "types.h"
#include "render.h"
#include "buffers.h"

#include "tetracut.h"

#include "denrender.h"

using namespace std;

//the depth of the triangle buffer
const int DenRender::VERTEXBUFFERDEPTH = 128 * 1024;

//must run in openGL environment, with glew

void DenRender::openGLInit(){
    fbuffer = new fluxBuffer(imagesize_, imagesize_);
    fbuffer->setBuffer();
    
    
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    
    // setup some generic opengl options
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_READ_COLOR_ARB, GL_FALSE);
    
    //setup blending
    glEnable (GL_BLEND);
    glBlendFunc (GL_ONE,GL_ONE);    //blending
    
    fbuffer->bindBuf();
    
    glPushAttrib(GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT);
    glViewport(0,0,imagesize_, imagesize_);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, viewSize, 0, viewSize, -100, +100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    //clear color
    glClearColor (0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    
    fbuffer->unbindBuf();
}



DenRender::DenRender(int imagesize, float boxsize,
                     float startz, float dz, int numplane,
                     int * argc, char * args[]){
    imagesize_ = imagesize;
    boxsize_ = boxsize;
    numplanes_ = numplane;
    
    image_ = new float[imagesize * imagesize * numplanes_];
    tempimage_ = new float[imagesize_ * imagesize_];
    
    //color and vertex
    vertexbuffer_ = new float[15 * VERTEXBUFFERDEPTH * numplanes_];
    vertexIds_ = new int[numplanes_];
    
    for(int i = 0; i < numplanes_; i ++){
        vertexIds_[i] = 0;
    }
    
    
    viewSize = boxsize_;
    
    argc_ = argc;
    args_ = args;
    
    startz_ = startz;
    dz_ = dz;
    
    //printf("debug1\n");
    openGLInit();
}

DenRender::~DenRender(){
    delete vertexIds_;
    delete vertexbuffer_;
    delete fbuffer;
    delete image_;
    delete tempimage_;
}

//render the i-th buffer
void DenRender::rendplane(int i){
    
    //printf("Rendering ... \n");
    
    //copy the data
    fbuffer->bindTex();
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, imagesize_, imagesize_, 0,
                 GL_RED, GL_FLOAT , 0);//image_ + i * imagesize_ * imagesize_);
    
    // set its parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    
    // setup some generic opengl options
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_READ_COLOR_ARB, GL_FALSE);
    
    fbuffer->unbindTex();
    
    //draw on them
    fbuffer->bindBuf();
    
    glPushAttrib(GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT);
    glViewport(0,0,imagesize_, imagesize_);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, viewSize, 0, viewSize, -100, +100);
    //glOrtho(-2, 2, -2, 2, -10, 10);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    //clear color
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    
    glEnableClientState (GL_VERTEX_ARRAY);
    glEnableClientState (GL_COLOR_ARRAY);
    
    glVertexPointer(2, GL_FLOAT, 5 * sizeof(GLfloat),
                     vertexbuffer_ + 15 * VERTEXBUFFERDEPTH * i);
    glColorPointer(3, GL_FLOAT, 5 * sizeof(GLfloat),
                     vertexbuffer_ + 15 * VERTEXBUFFERDEPTH * i + 2);
    
    glDrawArrays(GL_TRIANGLES, 0, vertexIds_[i] * 3);
    
    
    /*printf("Triangles: %d\n", i);
    float *p = vertexbuffer_ + 15 * VERTEXBUFFERDEPTH * i;
    printf("%f %f %e %e %e\n", *p, *(p+1), *(p+2), *(p+3), *(p+4));
    printf("%f %f %e %e %e\n", *(p+5), *(p+6), *(p+7), *(p+8), *(p+9));
    printf("%f %f %e %e %e\n", *(p+10), *(p+11), *(p+12), *(p+13), *(p+14));*/
    
    glFinish();
    fbuffer->unbindBuf();
    
    vertexIds_[i] = 0;
    
    //copy the data back
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    fbuffer->bindTex();
    glGetTexImage(GL_TEXTURE_2D,
                  0,
                  GL_RED,
                  GL_FLOAT,
                  tempimage_);//(image_ + imagesize_ * imagesize_ * i));
    fbuffer->unbindTex();
    
    //avoiding clamping
    float * imp = image_ + imagesize_ * imagesize_ * i;
    for(int j = 0; j < imagesize_ * imagesize_; j++){
        imp[j] += tempimage_[j];
    }
    //int j = 1000;
    //    printf("%e\n", *(image_ + imagesize_ * imagesize_ * i + j));
    
}

void DenRender::rend(Tetrahedron & tetra){
    cutter.setTetrahedron(&tetra);
    
    //printf("z = %f %f %f %f\n",
    //       tetra.v1.z, tetra.v2.z, tetra.v3.z, tetra.v4.z);
    
    if(startz_ > tetra.v4.z || tetra.v1.z > startz_ + numplanes_ * dz_){
        return;
    }
    
    int starti = max(floor((tetra.v1.z  - startz_ )/ dz_), 0.0f);
    int endi = min(ceil((tetra.v4.z  - startz_) / dz_), (float)numplanes_);
    
    //printf("starti endi = %f %d %f %d\n", startz_ + dz_ * starti, starti, startz_ + dz_ * endi, endi);
    //printf("allz = %f %f %f %f\n",
    //       tetra.v1.z, tetra.v2.z, tetra.v3.z, tetra.v4.z);
    
    for(int i = starti; i < endi; i++){
        float z = startz_ + dz_ * i;

        int tris = cutter.cut(z);
        for(int j = 0; j < tris; j++){
            
            /*printf("Triangles: \n");
            printf("%f %f\n", cutter.getTrangle(j).a.x, cutter.getTrangle(j).a.y);
            printf("%f %f\n", cutter.getTrangle(j).b.x, cutter.getTrangle(j).b.y);
            printf("%f %f\n", cutter.getTrangle(j).c.x, cutter.getTrangle(j).c.y);
            printf("%e %e\n",tetra.invVolume, tetra.volume);*/
            
            
            vertexbuffer_[15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15 + 0] = cutter.getTrangle(j).a.x;
            vertexbuffer_[15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15 + 1] = cutter.getTrangle(j).a.y;
            vertexbuffer_[15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15 + 2] = tetra.invVolume;
            vertexbuffer_[15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15 + 3] = tetra.invVolume;
            vertexbuffer_[15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15 + 4] = tetra.invVolume;
            vertexbuffer_[15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15 + 5] = cutter.getTrangle(j).b.x;
            vertexbuffer_[15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15 + 6] = cutter.getTrangle(j).b.y;
            vertexbuffer_[15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15 + 7] = tetra.invVolume;
            vertexbuffer_[15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15 + 8] = tetra.invVolume;
            vertexbuffer_[15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15 + 9] = tetra.invVolume;
            vertexbuffer_[15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15 + 10] = cutter.getTrangle(j).c.x;
            vertexbuffer_[15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15 + 11] = cutter.getTrangle(j).c.y;
            vertexbuffer_[15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15 + 12] = tetra.invVolume;
            vertexbuffer_[15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15 + 13] = tetra.invVolume;
            vertexbuffer_[15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15 + 14] = tetra.invVolume;

            
            /*printf("Triangles: %d\n", i);
            float *p = vertexbuffer_ + 15 * i * VERTEXBUFFERDEPTH + vertexIds_[i] * 15;
            printf("%f %f %e %e %e\n", *p, *(p+1), *(p+2), *(p+3), *(p+4));
            printf("%f %f %e %e %e\n", *(p+5), *(p+6), *(p+7), *(p+8), *(p+9));
            printf("%f %f %e %e %e\n", *(p+10), *(p+11), *(p+12), *(p+13), *(p+14));*/
            
            vertexIds_[i] ++;
            
            if(vertexIds_[i] >= VERTEXBUFFERDEPTH){
                rendplane(i);
                //printf("Rendering ... \n");
            }
        }
    }
}

float * DenRender::getDenfield(){
    return image_;
}


float * DenRender::getImage(){
    return image_;
}


void DenRender::finish(){
    for(int i = 0; i < numplanes_; i++){
        //printf("%d \n", vertexIds_[i]);
        rendplane(i);
    }
}

