#include <fstream>
#include <stdio.h>
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
    
    
    
    //test
    int totalTris = 0;
    
    
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

void TriDenRender::setOutputFile(string * outputfiles,
                                 int numOfOutputs
                                ){
    float startz = 0;
    float dz = boxsize_ / imagesize_;
    
    numOfOutputs_ = numOfOutputs;
    if(numOfOutputs_ > maxNumRenderComp){
        numOfOutputs_ = maxNumRenderComp;
    }
    
    outputStream_ = new fstream[numOfOutputs_];
    
    int head[59];
    for(int i = 0; i < numOfOutputs_; i++){
        outputStream_[i].open(outputfiles[i].c_str(), ios::out | ios::binary);
        if(!outputStream_[i].good()){
            //printf("%s\n", outputfiles[i].c_str());
            fprintf(stderr, "OutputFile Error: %s !\n", outputfiles[i].c_str());
            exit(1);
        }
        outputStream_[i].write((char *) &imagesize_, sizeof(int));
        outputStream_[i].write((char *) &imagesize_, sizeof(int));
        outputStream_[i].write((char *) &boxsize_, sizeof(float));
        outputStream_[i].write((char *) &startz, sizeof(float));
        outputStream_[i].write((char *) &dz, sizeof(float));
        outputStream_[i].write((char *) head, sizeof(int) * 59);
    }
}

TriDenRender::TriDenRender(int imagesize,
                           REAL boxSize,
                           string * outputfiles,
                           int numOfOutputs
                     ){

    imagesize_ = imagesize;

    

    boxsize_ = boxSize;
    result_ = new float[imagesize * imagesize * maxNumRenderComp];
    viewSize = boxSize;
    
    setOutputFile(outputfiles, numOfOutputs);
    init();
}

TriDenRender::~TriDenRender(){
    close();
    delete[] outputStream_;
    delete result_;
}

//void TriDenRender::rend(string verfile,
//                        string * componentFiles,
//                        int * floatPerTriangle){
void TriDenRender::rend(float * vertexdata,
                     float * densitydata,
                     int NumTriangles
                     ){
    if(numOfOutputs_ == 0){
        return;
    }
    
    memset(result_, 0, imagesize_ * imagesize_ * maxNumRenderComp * sizeof(float));
    
    //fstream verstream(verfile.c_str(), ios::in | ios::binary);
    
    //fstream * compStreams = new fstream[numOfOutputs_];
    //TriHeader header0;
    //verstream.read((char *) &header0, sizeof(header0));
    
    //TriHeader * header1 = new TriHeader[numOfOutputs_];

    //float * vertexdata = new float[header0.NumTriangles * NUM_FLOATS_VERTEX];

    
    //printf("ok2.5.1\n");
    
    float * colorData = new float[NumTriangles * 3 * maxNumRenderComp];
    

    /*verstream.read((char *) vertexdata,
                   sizeof(float) *
                   header0.NumTriangles *
                   NUM_FLOATS_VERTEX
                   );*/
    
    /*for(int i = 0; i < header0.numOfTriangles; i++){

    }*/
    
    
    /*float ** color0 = new float * [numOfOutputs_];
    for(int i = 0; i < numOfOutputs_; i++){
        fstream compStreams;
        compStreams.open(componentFiles[i].c_str(), ios::in | ios::binary);
        compStreams.read((char *) &(header1[i]), sizeof(TriHeader));
        if(header0.NumTriangles != header1[i].NumTriangles){
            fprintf(stderr, "Error, vertex and dens files does not match!\n");
            exit(1);
        }
        

        color0[i] = new float[header0.NumTriangles * floatPerTriangle[i]];

        
        compStreams.read((char *) (color0[i]),
                         sizeof(float) *
                         header0.NumTriangles *
                         floatPerTriangle[i]);
        compStreams.close();
    }*/

    
    //printf("ok2.5.2 -- %d\n", NumTriangles);
    for(unsigned int i = 0; i < NumTriangles; i++){
        //printf("ok2.5.2 -- %d\n", i);
        
        //test
        totalTris ++;
        
        for(int j = 0; j < maxNumRenderComp; j ++){
            if(j < numOfOutputs_){
                colorData[i * maxNumRenderComp * 3 + 0 + j] = densitydata[i];//color0[j][i * floatPerTriangle[j] + (floatPerTriangle[j] > 1 ? 0 : 0)];
                colorData[i * maxNumRenderComp * 3 + 4 + j] = densitydata[i];//color0[j][i * floatPerTriangle[j] + (floatPerTriangle[j] > 1 ? 1 : 0)];
                colorData[i * maxNumRenderComp * 3 + 8 + j] = densitydata[i];//color0[j][i * floatPerTriangle[j] + (floatPerTriangle[j] > 1 ? 2 : 0)];
            }else{
                colorData[i * maxNumRenderComp * 3 + 0 + j] = 0;
                colorData[i * maxNumRenderComp * 3 + 4 + j] = 0;
                colorData[i * maxNumRenderComp * 3 + 8 + j] = 0;
            }
        }
        
        /*printf("%f %f %f %f %f %f %e\n",
               vertexdata[i * NUM_FLOATS_VERTEX + 0],
               vertexdata[i * NUM_FLOATS_VERTEX + 1],
               vertexdata[i * NUM_FLOATS_VERTEX + 2],
               vertexdata[i * NUM_FLOATS_VERTEX + 3],
               vertexdata[i * NUM_FLOATS_VERTEX + 4],
               vertexdata[i * NUM_FLOATS_VERTEX + 5],
               colorData[i * maxNumRenderComp + 0]);*/
    }
    //printf("ok2.5.2.1\n");
    
    /*for(int j = 0; j < numOfOutputs_; j++){
        delete (color0[j]);
    }*/
    //delete color0;
    
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
                    GL_RGBA, GL_FLOAT, result_);
    /*glTexImage2D(GL_TEXTURE_2D,
                      0,
                      0,
                      imagesize_,
                      imagesize_,
                      0,
                      GL_RGBA,
                      GL_FLOAT,
                      NULL);*/
    //printf("ok2.5.2.1.2\n");
    
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
                  GL_RGBA,
                  GL_FLOAT,
                  result_);
    fbuffer->unbindTex();
    
    //printf("ok2.5.2.3\n");
    
    glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
    
    
    //printf("ok2.5.3\n");
    //save to file
    float ** res_ = new float *[numOfOutputs_];
    for(int i = 0; i < numOfOutputs_; i++){
        res_[i] = new float[imagesize_ * imagesize_];
    }
    
    for(int i = 0; i < imagesize_ * imagesize_; i++){
        for(int j = 0; j < numOfOutputs_; j++){
            res_[j][i] = result_[i * maxNumRenderComp + j];
        }
    }
    
    for(int j = 0; j < numOfOutputs_; j++){
        outputStream_[j].write((char *) res_[j],
                            sizeof(float) *
                            imagesize_ *
                            imagesize_);
        delete (res_[j]);
    }
    
    //printf("ok2.5.4\n");
    delete res_;
    delete colorData;
    //delete vertexdata;
}

bool TriDenRender::good(){
    bool isGood_ = false;
    for(int i = 0; i < numOfOutputs_; i++){
        isGood_ = isGood_ &&
                  outputStream_[i].good();
    }
    return isGood_;
}

void TriDenRender::close(){
    
    printf("Total Tris: %d\n", totalTris);
    for(int i = 0; i < numOfOutputs_; i++){
        outputStream_[i].close();
    }
}
