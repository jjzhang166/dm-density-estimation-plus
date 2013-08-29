#include <fstream>
#include "trirender.h"
#include "triheader.h"

#define NUM_FLOATS_VERTEX 7



namespace RenderSpace {
    
    buffer *fbuffer;
    REAL viewSize;
    
    //int num_of_rendertype = 0;
    
    //GLenum glFormats[] = {GL_RED, GL_RG, GL_RGB, GL_RGBA};
    
    static bool glut_is_initialized = false;
    
    
}

void TriRender::init(){
    
    
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



TriRender::TriRender(int imagesize,
                     string outputfile,
                     REAL boxSize
                     ){
    rentypes_ = rentypes;

    outputStream_.open(outputfilename[i].c_str(), ios::out | ios::bin);
    if(!outputStream_.good()){
        fprintf("OutputFile Error: %s !\n", outputfilename[i].c_str());
        exit(1);
    }
    
    result_ = new float[imagesize * imagesize];
    
    float startz = 0;
    float dz = boxSize / imagesize;
    int head[59]
    outputStream_.write((char *) &imagesize, sizeof(int));
    outputStream_.write((char *) &imagesize, sizeof(int));
    outputStream_.write((char *) &boxSize, sizeof(float));
    outputStream_.write((char *) &startz, sizeof(float));
    outputStream_.write((char *) &dz, sizeof(float));
    outputStream_.write((char *) head, sizeof(int) * 59);
    
    viewSize = boxSize;
    
    init();
}

TriRender::~TriRender(){
    outputStream_.close();
    delete result_;
}

void TriRender::rend(string planeFile){
    fstream inputFile(planeFile.c_str(), ios::in | ios::binary);
    TriHeader header;
    inputFile.read((char *) &header, sizeof(header));
    float * vertexdata = new float[header.numOfTriangles * NUM_FLOATS_VERTEX];
    float * colorData = new float[header.numOfTriangles * 3 * 3];
    for(int i = 0; i < header.numOfTriangles; i++){
        colorData[3 * i] = 
    }
    
    
}