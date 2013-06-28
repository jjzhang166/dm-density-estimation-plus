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


//must run in openGL environment, with glew

void DenRender::openGLInit(){
    //initialize glut and glew
    /*glutInit(argc_, argv_);
    glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);
    glutInitWindowSize(windowSize, windowSize);
    glutCreateWindow("Dark Matter Density rendering!");
    
#ifndef __APPLE__
    glewExperimental = GL_TRUE;
    glewInit();
#endif*/
    
    fbuffer = new fluxBuffer * [numplanes_];//(imagesize_, imagesize_);
    
    for(int i = 0; i < numplanes_; i++){
        fbuffer[i] = new fluxBuffer(imagesize_, imagesize_);
        fbuffer[i]->setBuffer();
    }
    
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    
    
    /*glGenTextures(1, &textureIni);
    glBindTexture(GL_TEXTURE_2D, textureIni);
    
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, windowSize, windowSize, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, 0);
    
    
    // set its parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    
    glBindTexture(GL_TEXTURE_2D, 0);
     */
    
    // setup some generic opengl options
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_READ_COLOR_ARB, GL_FALSE);
    
    //setup blending
    glEnable (GL_BLEND);
    glBlendFunc (GL_ONE,GL_ONE);    //blending
    
    for(int i = 0; i < numplanes_; i++){
        fbuffer[i]->bindBuf();
        
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
        
        fbuffer[i]->unbindBuf();
    }
}



DenRender::DenRender(int imagesize, float boxsize,
                     float startz, float dz, int numplane,
                     int * argc, char * args[]){
    imagesize_ = imagesize;
    boxsize_ = boxsize;
    numplanes_ = numplane;
    image_ = new float[imagesize * imagesize * numplane];
    
    viewSize = boxsize_;
    
    argc_ = argc;
    args_ = args;
    
    startz_ = startz;
    dz_ = dz;
    
    //printf("debug1\n");
    openGLInit();
}

DenRender::~DenRender(){
    for(int i = 0; i < numplanes_; i++){
        delete fbuffer[i];
    }
    delete fbuffer;
    delete image_;
}


void DenRender::rend(Tetrahedron & tetra){
    cutter.setTetrahedron(&tetra);
    if(startz_ > tetra.v4.z || tetra.v1.z > startz_ + numplanes_ * dz_){
        return;
    }
    
    for(int i = 0; i < numplanes_; i++){
        float z = startz_ + dz_ * i;
        int tris = cutter.cut(z);
        
        fbuffer[i]->bindBuf();
        
        glPushAttrib(GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT);
        glViewport(0,0,imagesize_, imagesize_);
        
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, viewSize, 0, viewSize, -100, +100);
        //glOrtho(-2, 2, -2, 2, -10, 10);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        
        //clear color
        glClearColor (0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glColor3f(tetra.invVolume, tetra.invVolume, tetra.invVolume);
        for(int j = 0; j < tris; j++){
            glBegin(GL_TRIANGLES); 
            glVertex3f(cutter.getTrangle(i).a.x,
                       cutter.getTrangle(i).a.y,
                       0.0f);

            glVertex3f(cutter.getTrangle(i).b.x,
                       cutter.getTrangle(i).b.y,
                       0.0f);
            
            glVertex3f(cutter.getTrangle(i).c.x,
                       cutter.getTrangle(i).c.y,
                       0.0f);
            glEnd();
        }
        
        fbuffer[i]->unbindBuf();
    }
    
}

float * DenRender::getDenfield(){
    for(int i = 0; i < numplanes_; i++){
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        fbuffer[i]->bindTex();
        glGetTexImage(GL_TEXTURE_2D,
                      0,
                      GL_RED,
                      GL_FLOAT,
                      (image_ + imagesize_ * imagesize_ * i));
        fbuffer[i]->unbindTex();
    }
    return image_;
}


float * DenRender::getImage(){
    return image_;
}

fluxBuffer ** DenRender::getBuffers(){
    return fbuffer;
}

