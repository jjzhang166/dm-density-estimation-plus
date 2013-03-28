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

using namespace std;

int windowSize;
fluxBuffer * fbuffer;
float viewSize;
GLuint textureIni;  //initial empty texture
int * argc;
char **argv;

void getWarmColor(float v, float &r, float &g, float &b){
    r = -1.8966 * v*v*v + 1.2049 * v*v + 1.1463 * v + 0.2253;
    g = -2.713 * v*v + 2.5221 * v + 0.2499;
    b = 2.3924 * v*v*v - 5.264 * v*v + 2.2936 * v + 0.7214;
    
    if(r > 1.0) r = 1.0;
    if(r < 0.0) r = 0.0;
    if(g > 1.0) g = 1.0;
    if(g < 0.0) g = 0.0;
    if(b > 1.0) b = 1.0;
    if(b < 0.0) b = 0.0;
}

void getJetColor(float value, float &r, float &g, float &b) {
    float fourValue = 4 * value;
    r = min(fourValue - 1.5, -fourValue + 4.5);
    g = min(fourValue - 0.5, -fourValue + 3.5);
    b = min(fourValue + 0.5, -fourValue + 2.5);
    if(r > 1.0) r = 1.0;
    if(r < 0.0) r = 0.0;
    if(g > 1.0) g = 1.0;
    if(g < 0.0) g = 0.0;
    if(b > 1.0) b = 1.0;
    if(b < 0.0) b = 0.0;
}

void getcolorImge(float *value, float * colorimg){
    float r, g, b;
    float max = 0.0, min = 1.0e20;
    for(int i = 0; i < windowSize * windowSize; i++){
        if(max < value[i]){
            max = value[i];
        }
        if(min > value[i] && value[i] > 0.0){
            min = value[i];
        }
    }
    
	//printf("%f %f\n", min, max);
	if(min == max){
		min = max / 1.0e5;	
	}

    float x = log(max) - log(min);
    for(int i = 0; i < windowSize * windowSize; i++){
        float v = (log(value[i]) - log(min)) / x;
        //printf("%f\n",v);
        getJetColor(v, r, g, b);
        colorimg[3 * i] = r;
        colorimg[3 * i + 1] = g;
        colorimg[3 * i + 2] = b;
    }
    
    glBindTexture(GL_TEXTURE_2D, textureIni);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, windowSize, windowSize, 0,
                 GL_RGB, GL_FLOAT, colorimg);
    // set its parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
}


void openGLInit(){
    //initialize glut and glew
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);
    glutInitWindowSize(windowSize, windowSize);
    glutCreateWindow("Dark Matter Density rendering!");
    
#ifndef __APPLE__
    glewExperimental = GL_TRUE;
    glewInit(); 
#endif

    fbuffer = new fluxBuffer(windowSize, windowSize);
    fbuffer->setBuffer();

    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);

    
    glGenTextures(1, &textureIni);
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
    
    // setup some generic opengl options
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_READ_COLOR_ARB, GL_FALSE);
    
    //setup blending
    glEnable (GL_BLEND);
    glBlendFunc (GL_ONE,GL_ONE);    //blending
}



Render::Render(int imagesize, REAL boxsize, TetraIsoPlane * isoplane,
               int * argc_, char * args_[]){
    imagesize_ = imagesize;
    windowSize = imagesize_;
    boxsize_ = boxsize;
    isoplane_ = isoplane;
    image_ = new float[imagesize * imagesize];
    colorImg_ = new float[imagesize * imagesize * 3];
    viewSize = boxsize_;
    argc = argc_;
    argv = args_;
    //printf("debug1\n");
    openGLInit();
    
}

double Render::getRenderTime(){
    return rendertime_;
}


float * Render::getPlane(REAL isoval){
    timeval timediff;
    double t1, t0 = 0;

    isoplane_ -> setIsoValue(isoval);
    fbuffer->bindBuf();
    
    glPushAttrib(GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT);
    glViewport(0,0,windowSize, windowSize);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, viewSize, 0, viewSize, -100, +100);
    //glOrtho(-2, 2, -2, 2, -10, 10);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    //clear color
    glClearColor (0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
    
    glEnableClientState (GL_VERTEX_ARRAY);
    glEnableClientState (GL_COLOR_ARRAY);
    
    rendertime_ = 0;
    for(int i = 0; i < isoplane_->getTotalBlockNum(); i++){
        
        isoplane_->loadIsoplane(i);
        int num_triangles;
        while(isoplane_-> hasNext()){
            Triangle * triangles = isoplane_->getNextIsoPlaneBlock(num_triangles);
            //printf("%d\n",num_triangles);
            GLfloat * vetexarray = (GLfloat *) triangles;
            glVertexPointer (2, GL_FLOAT, 5 * sizeof(GLfloat), &(vetexarray[0]));
            glColorPointer (3, GL_FLOAT, 5 * sizeof(GLfloat), &(vetexarray[2]));
  
            gettimeofday(&timediff, NULL);
            t0 = timediff.tv_sec + timediff.tv_usec / 1.0e6;

            glDrawArrays(GL_TRIANGLES, 0, num_triangles * 3);

            gettimeofday(&timediff, NULL);
	        t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
            rendertime_ += (t1 - t0);
        }
    }

    gettimeofday(&timediff, NULL);
    t0 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
    
    glFinish();
    
    gettimeofday(&timediff, NULL);
	t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
    rendertime_ += (t1 - t0);

	/*glBegin(GL_TRIANGLES);
    glColor3f(0.3, 0.0,0);
    glVertex3f(0, 0, 0);
    glVertex3f(viewSize, 0, 0);
    glVertex3f(0, viewSize, 0);
    glVertex3f(10402.908203, -235.477020,0);
    glVertex3f(viewSize/2, -viewSize/2,0);
    glVertex3f(0, viewSize/2,0);
    //glVertex3f(-1, -1,0);
    //glVertex3f(1, -1 ,0);
    //glVertex3f(0, 1, 0);
    glEnd();*/
   
    glDisableClientState (GL_VERTEX_ARRAY);
    glDisableClientState (GL_COLOR_ARRAY);

    fbuffer->unbindBuf();
    
    //getback the image
    glPixelStorei(GL_PACK_ALIGNMENT, 4);  
    fbuffer->bindTex();
    glGetTexImage(GL_TEXTURE_2D,0,GL_RED,GL_FLOAT,image_);

	//for(int i = 0; i < 512; i++){
	//	printf("%e\n", image_[i]);	
	//}

    fbuffer->unbindTex();
    return image_;
    
}



void ReshapeFunc(int width, int height)
{
    glViewport(0, 0, width, height);
}


void KeyboardFunc(unsigned char key, int x, int y)
{
    //int foo;
    //foo = x + y; //Has no effect: just to avoid a warning
    if ('q' == key || 'Q' == key || 27 == key){
        exit(0);
    //else if(key == 's' && picfile!=""){
        //savePic();
    }
}


void rendsenc(){
    //glClientActiveTexture(GL_TEXTURE0);
	glViewport(0,0,windowSize, windowSize);
    glActiveTexture(GL_TEXTURE0);
    
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_POINT_SPRITE);
    glEnable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);
    
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2, 2, -2, 2, -100, 100);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    //fbuffer -> bindTex();
    //printf(">>>\n");
    glBindTexture(GL_TEXTURE_2D, textureIni);
    
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0); glVertex3f(-2, -2, 0);
    glTexCoord2i(0, 1); glVertex3f(-2,  2, 0);
    glTexCoord2i(1, 1); glVertex3f( 2,  2, 0);
    glTexCoord2i(1, 0); glVertex3f( 2, -2, 0);
    glEnd();
    
	/*glBindTexture(GL_TEXTURE_2D, 0);
    glBegin(GL_TRIANGLES);
    glColor3f(1,0,0);
    glVertex3f(0,1,0);
    glVertex3f(-1,0,0);
    glVertex3f(1,0,0);
    glEnd();*/
    
    glBindTexture(GL_TEXTURE_2D, 0);
    glFlush();
}

float * Render::showPlane(REAL isoval){
    printf("Start rendering ...\n");
    timeval timediff;
    double t1, t0, t_total = 0;

    gettimeofday(&timediff, NULL);
    t0 = timediff.tv_sec + timediff.tv_usec / 1.0e6;

    getPlane(isoval);
    getcolorImge(image_, colorImg_);

    gettimeofday(&timediff, NULL);
    t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
    t_total = t1 - t0;
    printf("Total time %f, rendering time %f, uses %f%% for rendering.\n", 
                    t_total, rendertime_, rendertime_/t_total * 100);
    
    printf("Cutting time %f, uses %f%% for cutting.\n", isoplane_->getCutTime(),
           isoplane_->getCutTime()/t_total*100);
    
    printf("Datareading time %f, uses %f%% for reading.\n", isoplane_->getIndStream()->getRunningTime(),
           isoplane_->getIndStream()->getRunningTime()/t_total*100);

    
    //set up glut
    glutDisplayFunc(&rendsenc);
    glutReshapeFunc(&ReshapeFunc);
    glutKeyboardFunc(&KeyboardFunc);
    glutMainLoop();
    return image_;
}

Render::~Render(){
    delete fbuffer;
    delete image_;
    delete colorImg_;
}
