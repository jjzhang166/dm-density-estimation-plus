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
    
    //GLenum glFormats[] = {GL_RED, GL_RG, GL_RGB, GL_RGBA};
    
    static bool glut_is_initialized = false;
    
    
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



TriDenRender::TriDenRender(int imagesize,
                     string outputfile,
                     REAL boxSize
                     ){

    imagesize_ = imagesize;
    outputStream_.open(outputfile.c_str(), ios::out | ios::binary);
    if(!outputStream_.good()){
        fprintf(stderr, "OutputFile Error: %s !\n", outputfile.c_str());
        exit(1);
    }
    
    result_ = new float[imagesize * imagesize];
    
    float startz = 0;
    float dz = boxSize / imagesize;
    boxsize_ = boxSize;
    int head[59];
    outputStream_.write((char *) &imagesize, sizeof(int));
    outputStream_.write((char *) &imagesize, sizeof(int));
    outputStream_.write((char *) &boxSize, sizeof(float));
    outputStream_.write((char *) &startz, sizeof(float));
    outputStream_.write((char *) &dz, sizeof(float));
    outputStream_.write((char *) head, sizeof(int) * 59);
    
    viewSize = boxSize;
    
    init();
}

TriDenRender::~TriDenRender(){
    outputStream_.close();
    delete result_;
}

void TriDenRender::rend(string verfile, string denfile){
    memset(result_, 0, imagesize_ * imagesize_ * sizeof(float));
    /*for(int i = 0; i < imagesize_ * imagesize_; i++){
        result_[i] = 0;
    }*/
    
    fstream verstream(verfile.c_str(), ios::in | ios::binary);
    fstream denstream(denfile.c_str(), ios::in | ios::binary);
    TriHeader header0, header1;
    verstream.read((char *) &header0, sizeof(header0));
    denstream.read((char *) &header1, sizeof(header1));
    if(header0.numOfTriangles != header1.numOfTriangles){
        fprintf(stderr, "Error, vertex and dens files does not match!\n");
        exit(1);
    }
    
    float * vertexdata = new float[header0.numOfTriangles * NUM_FLOATS_VERTEX];
    float * color0 = new float[header0.numOfTriangles];
    float * colorData = new float[header0.numOfTriangles * 3 * 3];
    
    verstream.read((char *) vertexdata,
                   sizeof(float) * header0.numOfTriangles * NUM_FLOATS_VERTEX
                   );
    denstream.read((char *) color0, sizeof(float) * header0.numOfTriangles);
    
    //test
    //printf("Num of Tris: %d\n", header0.numOfTriangles);
    
    for(int i = 0; i < header0.numOfTriangles; i++){
        for(int j = 0; j < 9; j ++){
            colorData[9 * i + j] = color0[i];
           
        }
        /*printf("%f %f %f %f %f %f %e \n",
               vertexdata[i * NUM_FLOATS_VERTEX + 0],
               vertexdata[i * NUM_FLOATS_VERTEX + 1],
               vertexdata[i * NUM_FLOATS_VERTEX + 2],
               vertexdata[i * NUM_FLOATS_VERTEX + 3],
               vertexdata[i * NUM_FLOATS_VERTEX + 4],
               vertexdata[i * NUM_FLOATS_VERTEX + 5],
               color0[i]);*/
    }
    delete color0;
    
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
    
    
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imagesize_, imagesize_,
                    GL_RED, GL_FLOAT , result_);
    
    
    fbuffer->unbindTex();
    
    //draw on them
    fbuffer->bindBuf();
    
    //clear the scene
    //glClearColor(0, 0, 0, 0);

    
    
    glPushAttrib(GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT);
    glViewport(0,0,imagesize_, imagesize_);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, viewSize, 0, viewSize, -100, +100);
    //glOrtho(-2, 2, -2, 2, -10, 10);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    //clear color
    //glClearColor(0, 0, 0, 0);
    

    glEnableClientState (GL_VERTEX_ARRAY);
    glEnableClientState (GL_COLOR_ARRAY);
    
    glVertexPointer(2, GL_FLOAT, sizeof(float) * 2,
                    vertexdata);
    glColorPointer(3, GL_FLOAT, sizeof(float) * 3,
                    colorData);
    
    glDrawArrays(GL_TRIANGLES, 0, header0.numOfTriangles * 3);
    
    
    glFinish();
    fbuffer->unbindBuf();
    
    //copy the data back
    //glPixelStorei(GL_PACK_ALIGNMENT, 4);
    fbuffer->bindTex();
    glGetTexImage(GL_TEXTURE_2D,
                  0,
                  GL_RED,
                  GL_FLOAT,
                  //tempimage_);
                  (result_));
    fbuffer->unbindTex();
    
    //save to file
    outputStream_.write((char *) result_, sizeof(float) * imagesize_ * imagesize_);
    
    delete colorData;
    delete vertexdata;
}

bool TriDenRender::good(){
    return outputStream_.good();
}

void TriDenRender::close(){
    outputStream_.close();
}
