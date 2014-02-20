#include <fstream>
#include <stdio.h>
#include <cstdlib>
#include <string.h>
#ifdef __APPLE__
#include <GLUT/glut.h> // darwin uses glut.h rather than GL/glut.h
#else
#include <GL/glew.h>
#include <GL/glut.h>
#endif
#include "types.h"
#include "buffers.h"
#include "trirender.h"
#include "triheader.h"

#define NUM_FLOATS_VERTEX 6

namespace RenderSpace {
    
    buffer *fbuffer;
    REAL viewSize;
    
    //int num_of_rendertype = 0;
    
    GLenum glFormats[] = {GL_RED, GL_RG, GL_RGB, GL_RGBA};
    
    static bool glut_is_initialized = false;
    
    
    int currentAllocedColorSize = 0;
    //test
    //int totalTris = 0;
    
    
}
using namespace RenderSpace;

void TriDenRender::init(){
    
    
    if(!glut_is_initialized){
        
        
        int argv = 1;
        char * args[1];
        args[0] = (char *) "LTFE Render";
        
        glutInit(&argv, args);
        glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);
        glutInitWindowSize(imagesize_, imagesize_);
        glutCreateWindow("Dark Matter Density rendering!");
        
#ifndef __APPLE__
        glewExperimental = GL_TRUE;
        glewInit();
#endif
        glut_is_initialized = true;
        
    }
    fbuffer = new buffer(imagesize_, imagesize_);
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


const int TriDenRender::maxNumRenderComp = 4;


TriDenRender::TriDenRender(int imagesize,
                           REAL boxSize
                           ){
    
    imagesize_ = imagesize;
    
    
    
    boxsize_ = boxSize;
    density_ = new float[imagesize * imagesize];
    velocityx_ = new float[imagesize * imagesize];
    velocityy_ = new float[imagesize * imagesize];
    velocityz_ = new float[imagesize * imagesize];
    
    currentAllocedColorSize = imagesize * imagesize * 6;
    colorData = (float *) malloc(currentAllocedColorSize * sizeof(float));
    
    viewSize = boxSize;
    
    //setOutputFile(outputfiles, numOfOutputs);
    init();
}

TriDenRender::~TriDenRender(){
    //close();
    //delete[] outputStream_;
    //delete[] result_;
    delete[] density_;
    delete[] velocityx_;
    delete[] velocityy_;
    delete[] velocityz_;
    free(colorData);
    
    //test
    //printf("Total Tris: %d\n", totalTris);
}

void TriDenRender::rendDensity(float * vertexdata,
                     float * densitydata,
                     int NumTriangles,
                     bool isClear
                     ){
    
    if(isClear){
        memset(density_, 0, imagesize_ * imagesize_ * sizeof(float));
    }
    
    int newColorSize = NumTriangles * 3 * maxNumRenderComp;
    if(currentAllocedColorSize < newColorSize){
        currentAllocedColorSize = newColorSize * 2;
        colorData = (float *) realloc(colorData, currentAllocedColorSize * sizeof(float));//new float[NumTriangles * 3 * maxNumRenderComp];
        //printf("ok\n");
    }

    
    //printf("ok2.5.2 -- %d\n", NumTriangles);
    for(int i = 0; i < NumTriangles; i++){
        //printf("ok2.5.2 -- %d\n", i);
        
        //test
        //totalTris ++;
        
        for(int j = 0; j < maxNumRenderComp; j ++){
            if(j < 1){
                colorData[i * maxNumRenderComp * 3 + 0 + j] = densitydata[i];//color0[j][i * floatPerTriangle[j] + (floatPerTriangle[j] > 1 ? 0 : 0)];
                colorData[i * maxNumRenderComp * 3 + 4 + j] = densitydata[i];//color0[j][i * floatPerTriangle[j] + (floatPerTriangle[j] > 1 ? 1 : 0)];
                colorData[i * maxNumRenderComp * 3 + 8 + j] = densitydata[i];//color0[j][i * floatPerTriangle[j] + (floatPerTriangle[j] > 1 ? 2 : 0)];
            }else{
                colorData[i * maxNumRenderComp * 3 + 0 + j] = 0;
                colorData[i * maxNumRenderComp * 3 + 4 + j] = 0;
                colorData[i * maxNumRenderComp * 3 + 8 + j] = 0;
            }
        }
        
    }

    rend(NumTriangles, vertexdata);
}



void TriDenRender::rendDensity(float * vertexdata,
                               float * densitydata,
                               float * velxdata,
                               float * velydata,
                               float * velzdata,
                               int NumTriangles,
                               bool isDisp,
                               bool isClear
                               ){
    
    if(isClear){
        memset(density_, 0, imagesize_ * imagesize_ * sizeof(float));
        memset(velocityx_, 0, imagesize_ * imagesize_ * sizeof(float));
        memset(velocityy_, 0, imagesize_ * imagesize_ * sizeof(float));
        memset(velocityz_, 0, imagesize_ * imagesize_ * sizeof(float));
    }
    
    int newColorSize = NumTriangles * 3 * maxNumRenderComp;
    if(currentAllocedColorSize < newColorSize){
        currentAllocedColorSize = newColorSize * 2;
        colorData = (float *) realloc(colorData, currentAllocedColorSize * sizeof(float));//new float[NumTriangles * 3 * maxNumRenderComp];
        //printf("ok\n");
    }
    
    
    //printf("ok2.5.2 -- %d\n", NumTriangles);
    for(int i = 0; i < NumTriangles; i++){

        colorData[i * maxNumRenderComp * 3 + 0 + 0] = densitydata[i];
        colorData[i * maxNumRenderComp * 3 + 4 + 0] = densitydata[i];
        colorData[i * maxNumRenderComp * 3 + 8 + 0] = densitydata[i];
        
        if(!isDisp){
            // velocity field: \sum rho_i v_i
            colorData[i * maxNumRenderComp * 3 + 0 + 1] = velxdata[i * 3 + 0];
            colorData[i * maxNumRenderComp * 3 + 0 + 2] = velydata[i * 3 + 0];
            colorData[i * maxNumRenderComp * 3 + 0 + 3] = velzdata[i * 3 + 0];
            
            colorData[i * maxNumRenderComp * 3 + 4 + 1] = velxdata[i * 3 + 1];
            colorData[i * maxNumRenderComp * 3 + 4 + 2] = velydata[i * 3 + 1];
            colorData[i * maxNumRenderComp * 3 + 4 + 3] = velzdata[i * 3 + 1];
            
            colorData[i * maxNumRenderComp * 3 + 8 + 1] = velxdata[i * 3 + 2];
            colorData[i * maxNumRenderComp * 3 + 8 + 2] = velydata[i * 3 + 2];
            colorData[i * maxNumRenderComp * 3 + 8 + 3] = velzdata[i * 3 + 2];
        }else{
            //velocity dispersion: \sum rho_i v_i^2
            colorData[i * maxNumRenderComp * 3 + 0 + 1] = velxdata[i * 3 + 0] * velxdata[i * 3 + 0] / densitydata[i];
            colorData[i * maxNumRenderComp * 3 + 0 + 2] = velydata[i * 3 + 0] * velydata[i * 3 + 0] / densitydata[i];
            colorData[i * maxNumRenderComp * 3 + 0 + 3] = velzdata[i * 3 + 0] * velzdata[i * 3 + 0] / densitydata[i];
            
            colorData[i * maxNumRenderComp * 3 + 4 + 1] = velxdata[i * 3 + 1] * velxdata[i * 3 + 1] / densitydata[i];
            colorData[i * maxNumRenderComp * 3 + 4 + 2] = velydata[i * 3 + 1] * velydata[i * 3 + 1] / densitydata[i];
            colorData[i * maxNumRenderComp * 3 + 4 + 3] = velzdata[i * 3 + 1] * velzdata[i * 3 + 1] / densitydata[i];
            
            colorData[i * maxNumRenderComp * 3 + 8 + 1] = velxdata[i * 3 + 2] * velxdata[i * 3 + 2] / densitydata[i];
            colorData[i * maxNumRenderComp * 3 + 8 + 2] = velydata[i * 3 + 2] * velydata[i * 3 + 2] / densitydata[i];
            colorData[i * maxNumRenderComp * 3 + 8 + 3] = velzdata[i * 3 + 2] * velzdata[i * 3 + 2] / densitydata[i];
        }
        
    }

    
    rend(NumTriangles, vertexdata);
}

void TriDenRender::rend(int NumTriangles, float * vertexdata){
    
    
    //copy the data
    fbuffer->bindTex();
    
    // setup some generic opengl options
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_READ_COLOR_ARB, GL_FALSE);
    
    
    // set its parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    
    
    //printf("ok2.5.2.1.1\n");
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imagesize_, imagesize_,
                    //GL_RGBA,
                    GL_RED,
                    GL_FLOAT, density_);
    
    
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imagesize_, imagesize_,
                    //GL_RGBA,
                    GL_GREEN,
                    GL_FLOAT, velocityx_);
    
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imagesize_, imagesize_,
                    //GL_RGBA,
                    GL_BLUE,
                    GL_FLOAT, velocityy_);
    
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imagesize_, imagesize_,
                    //GL_RGBA,
                    GL_ALPHA,
                    GL_FLOAT, velocityz_);
    
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
    glEnableClientState (GL_VERTEX_ARRAY);
    glEnableClientState (GL_COLOR_ARRAY);
    
    //printf("ok2.5.2.2\n");
    
    glVertexPointer(2, GL_FLOAT, 0,
                    vertexdata);
    glColorPointer(4, GL_FLOAT, 0,
                   colorData);
    
    glDrawArrays(GL_TRIANGLES, 0,
                 NumTriangles * 3);
    
    
    glFinish();
    fbuffer->unbindBuf();
    
    //copy the data back
    //glPixelStorei(GL_PACK_ALIGNMENT, 4);
    fbuffer->bindTex();
    glGetTexImage(GL_TEXTURE_2D,
                  0,
                  //GL_RGBA,
                  GL_RED,
                  GL_FLOAT,
                  density_);
    
    glGetTexImage(GL_TEXTURE_2D,
                  0,
                  //GL_RGBA,
                  GL_GREEN,
                  GL_FLOAT,
                  velocityx_);
    
    glGetTexImage(GL_TEXTURE_2D,
                  0,
                  //GL_RGBA,
                  GL_BLUE,
                  GL_FLOAT,
                  velocityy_);
    
    glGetTexImage(GL_TEXTURE_2D,
                  0,
                  //GL_RGBA,
                  GL_ALPHA,
                  GL_FLOAT,
                  velocityz_);
    fbuffer->unbindTex();
    
    //printf("ok2.5.2.3\n");
    
    glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}





float * TriDenRender::getDensity(){
    return density_;
}

float * TriDenRender::getVelocityX(){
    return velocityx_;
}

float * TriDenRender::getVelocityY(){
    return velocityy_;
}

float * TriDenRender::getVelocityZ(){
    return velocityz_;
}