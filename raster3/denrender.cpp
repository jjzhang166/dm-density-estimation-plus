#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef __APPLE__
//#include <glew.h>
#include <GLUT/glut.h> // darwin uses glut.h rather than GL/glut.h
#else
#include <GL/glew.h>
#include <GL/glut.h>
//#include <GL/glx.h>
#endif

#include <cmath>

#if defined(_WIN32) || defined(_WIN64)
//#include "gettimeofday_win.h"
#else
#include <unistd.h>
#include <sys/time.h>
#endif


#include "types.h"
#include "buffers.h"

#include "tetracut.h"

#include "denrender.h"

//number floats of a triangle
#define NUM_FLOATS_TRIANGLE 18

using namespace std;


//test
int totalTris = 0;


namespace RenderSpace {
    //buffer *fbuffer;
    //the buffer for drawing triangles
    float * vertexbuffer_;
    int * vertexIds_;
    
    buffer *fbuffer;
    REAL viewSize;
    
    int num_of_rendertype = 0;
    
    GLenum glFormats[] = {GL_RED, GL_RG, GL_RGB, GL_RGBA};
    
    static bool glut_is_initialized = false;
    
    
}

//the depth of the triangle buffer
const int DenRender::VERTEXBUFFERDEPTH = 64 * 1024;

//at most render 4 different render type at the same time
const int DenRender::NUM_OF_RENDERTRYPE_LIMIT = 4;


using namespace RenderSpace;

//must run in openGL environment, with glew

void DenRender::init(){

    
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
        //delete args[0];
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



DenRender::DenRender(int imagesize,
                     float boxsize,
                     float startz,
                     float dz,
                     int numplane,
                     vector<RenderType> rentypes){
    
    imagesize_ = imagesize;
    boxsize_ = boxsize;
    numplanes_ = numplane;
    
    rentypes_ = rentypes;
    num_of_rendertype = rentypes.size();
    
    if(num_of_rendertype > NUM_OF_RENDERTRYPE_LIMIT){
        printf("Number of render components larger than limit %d.\n", NUM_OF_RENDERTRYPE_LIMIT);
        printf("Only first %d render components will be calculated.\n", NUM_OF_RENDERTRYPE_LIMIT);
        num_of_rendertype = NUM_OF_RENDERTRYPE_LIMIT;
    }
    
    
    result_ = new float[imagesize * imagesize * numplanes_ * num_of_rendertype];
    
    for(int i = 0; i < imagesize * imagesize * numplanes_ * num_of_rendertype; i++){
        result_[i] = 0.0f;
    }
    
    //color and vertex
    vertexbuffer_ = new float[NUM_FLOATS_TRIANGLE * VERTEXBUFFERDEPTH * numplanes_];
    vertexIds_ = new int[numplanes_];
    
    for(int i = 0; i < numplanes_; i ++){
        vertexIds_[i] = 0;
    }
    
    
    viewSize = boxsize_;
    
    
    startz_ = startz;
    dz_ = dz;
    
    
    init();
}

DenRender::~DenRender(){
    delete[] vertexIds_;
    delete[] vertexbuffer_;
    delete fbuffer;
    delete[] result_;
}

//render the i-th buffer
void DenRender::rendplane(int i){
    
    
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
                    glFormats[num_of_rendertype - 1], GL_FLOAT , result_ +
                    i * imagesize_ * imagesize_ * num_of_rendertype);

    
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
    
    glVertexPointer(2, GL_FLOAT, 6 * sizeof(GLfloat),
                     vertexbuffer_ + NUM_FLOATS_TRIANGLE * VERTEXBUFFERDEPTH * i);
    glColorPointer(4, GL_FLOAT, 6 * sizeof(GLfloat),
                     vertexbuffer_ + NUM_FLOATS_TRIANGLE * VERTEXBUFFERDEPTH * i + 2);
    
    glDrawArrays(GL_TRIANGLES, 0, vertexIds_[i] * 3);
    
    
    glFinish();
    fbuffer->unbindBuf();
    
    vertexIds_[i] = 0;
    
    //copy the data back
    //glPixelStorei(GL_PACK_ALIGNMENT, 4);
    fbuffer->bindTex();
    glGetTexImage(GL_TEXTURE_2D,
                  0,
                  glFormats[num_of_rendertype - 1],
                  GL_FLOAT,
                  //tempimage_);
                  (result_ + imagesize_ * imagesize_ * i * num_of_rendertype));
    fbuffer->unbindTex();
    
}

void DenRender::rend(Tetrahedron & tetra){
    cutter.setTetrahedron(&tetra);
    
    //printf("z = %f %f %f %f\n",
    //       tetra.v1.z, tetra.v2.z, tetra.v3.z, tetra.v4.z);
    
   // printf("%f %f\n", startz_, dz_);
    
    if(startz_ > tetra.v4.z || tetra.v1.z > startz_ + numplanes_ * dz_){
        return;
    }
    
    int starti = max(floor((tetra.v1.z  - startz_ )/ dz_), 0.0f);
    int endi = min(ceil((tetra.v4.z  - startz_) / dz_), (float)numplanes_);
    
    for(int i = starti; i < endi; i++){
        float z = startz_ + dz_ * i;

        int tris = cutter.cut(z);
        
        
        for(int j = 0; j < tris; j++){
            //test
            totalTris ++ ;
            
            
            //density, stream number, velocity_x, velocity_y, velocity_z
            
            float values_a[] = {tetra.invVolume, 1,
                cutter.getTriangle(j).val1.x * tetra.invVolume,
                cutter.getTriangle(j).val1.y * tetra.invVolume,
                cutter.getTriangle(j).val1.z * tetra.invVolume
            };
            
            float values_b[] = {tetra.invVolume, 1,
                cutter.getTriangle(j).val2.x * tetra.invVolume,
                cutter.getTriangle(j).val2.y * tetra.invVolume,
                cutter.getTriangle(j).val2.z * tetra.invVolume
            };
            
            float values_c[] = {tetra.invVolume, 1,
                cutter.getTriangle(j).val3.x * tetra.invVolume,
                cutter.getTriangle(j).val3.y * tetra.invVolume,
                cutter.getTriangle(j).val3.z * tetra.invVolume
            };
            
            vertexbuffer_[NUM_FLOATS_TRIANGLE * i * VERTEXBUFFERDEPTH
                          + vertexIds_[i] * NUM_FLOATS_TRIANGLE
                          + 0] = cutter.getTriangle(j).a.x;
            
            vertexbuffer_[NUM_FLOATS_TRIANGLE * i * VERTEXBUFFERDEPTH
                          + vertexIds_[i] * NUM_FLOATS_TRIANGLE
                          + 1] = cutter.getTriangle(j).a.y;
            
            for(int k = 0; k < NUM_OF_RENDERTRYPE_LIMIT; k ++ ){
                if(k < num_of_rendertype){
                    vertexbuffer_[NUM_FLOATS_TRIANGLE * i * VERTEXBUFFERDEPTH
                                  + vertexIds_[i] * NUM_FLOATS_TRIANGLE
                                  + 2 + k] = values_a[rentypes_[k]];

                }else{
                    vertexbuffer_[NUM_FLOATS_TRIANGLE * i * VERTEXBUFFERDEPTH
                                  + vertexIds_[i] * NUM_FLOATS_TRIANGLE
                                  + 2 + k] = 0.0;
                }
            }
            

            vertexbuffer_[NUM_FLOATS_TRIANGLE * i * VERTEXBUFFERDEPTH
                          + vertexIds_[i] * NUM_FLOATS_TRIANGLE
                          + 6] = cutter.getTriangle(j).b.x;
            
            vertexbuffer_[NUM_FLOATS_TRIANGLE * i * VERTEXBUFFERDEPTH
                          + vertexIds_[i] * NUM_FLOATS_TRIANGLE
                          + 7] = cutter.getTriangle(j).b.y;
            
            for(int k = 0; k < NUM_OF_RENDERTRYPE_LIMIT; k ++ ){
                if(k < num_of_rendertype){
                    vertexbuffer_[NUM_FLOATS_TRIANGLE * i * VERTEXBUFFERDEPTH
                                  + vertexIds_[i] * NUM_FLOATS_TRIANGLE
                                  + 8 + k] = values_b[rentypes_[k]];
                    
                }else{
                    vertexbuffer_[NUM_FLOATS_TRIANGLE * i * VERTEXBUFFERDEPTH
                                  + vertexIds_[i] * NUM_FLOATS_TRIANGLE
                                  + 8 + k] = 0.0;
                }
            }
            
            
            
            vertexbuffer_[NUM_FLOATS_TRIANGLE * i * VERTEXBUFFERDEPTH
                          + vertexIds_[i] * NUM_FLOATS_TRIANGLE
                          + 12] = cutter.getTriangle(j).c.x;
            
            vertexbuffer_[NUM_FLOATS_TRIANGLE * i * VERTEXBUFFERDEPTH
                          + vertexIds_[i] * NUM_FLOATS_TRIANGLE
                          + 13] = cutter.getTriangle(j).c.y;
            
            for(int k = 0; k < NUM_OF_RENDERTRYPE_LIMIT; k ++ ){
                if(k < num_of_rendertype){
                    vertexbuffer_[NUM_FLOATS_TRIANGLE * i * VERTEXBUFFERDEPTH
                                  + vertexIds_[i] * NUM_FLOATS_TRIANGLE
                                  + 14 + k] = values_c[rentypes_[k]];
                    
                }else{
                    vertexbuffer_[NUM_FLOATS_TRIANGLE * i * VERTEXBUFFERDEPTH
                                  + vertexIds_[i] * NUM_FLOATS_TRIANGLE
                                  + 14 + k] = 0.0;
                }
            }
            
            
            vertexIds_[i] ++;
            
            if(vertexIds_[i] >= VERTEXBUFFERDEPTH){
                rendplane(i);
                //printf("Rendering ... \n");
            }
        }
    }
}


float * DenRender::getResult(){
    
    printf("Total Tris: %d\n", totalTris);
    return result_;
}

void DenRender::finish(){
    
    for(int i = 0; i < numplanes_; i++){
        //printf("%d \n", vertexIds_[i]);
        if(vertexIds_[i] > 0){
            rendplane(i);
        }
    }
    
    //for(int i = 0; i < imagesize_ * imagesize_ * numplanes_; i++){
    //    density_[i] = image_[i * NUMCOLORCOMP];
    //    streams_[i] = image_[i * NUMCOLORCOMP + 1];
    //}
}

