/****************************************************/
/* View the density*/
/****************************************************/

#include <fstream>
#include <cstdio>

#ifdef __APPLE__
#include <GLUT/glut.h> // darwin uses glut.h rather than GL/glut.h
#else
#include <GL/glew.h>
#include <GL/glut.h>
#endif


#include "viewfun.h"

using namespace std;


GLuint textureIni;
float * colorim;
int gridsize = 0;
int numofcuts = 0;
int pti = 0;

void rendsenc(){
    //glClientActiveTexture(GL_TEXTURE0);
	glViewport(0,0,gridsize, gridsize);
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
    //printf("Texture: %d\n", pti);
    //glBindTexture(GL_TEXTURE_2D, textureIni[pti]);
    
    glBindTexture(GL_TEXTURE_2D, textureIni);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, gridsize, gridsize, 0,
                 GL_RGB, GL_FLOAT, colorim + pti * 3 * gridsize * gridsize);
    // set its parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    
    
    glBegin(GL_QUADS);
    glTexCoord2i(0, 0); glVertex3f(-2, -2, 0);
    glTexCoord2i(0, 1); glVertex3f(-2,  2, 0);
    glTexCoord2i(1, 1); glVertex3f( 2,  2, 0);
    glTexCoord2i(1, 0); glVertex3f( 2, -2, 0);
    glEnd();
    
    glBindTexture(GL_TEXTURE_2D, 0);
    glFlush();
}


void KeyboardFunc(unsigned char key, int x, int y)
{
    //int foo;
    //foo = x + y; //Has no effect: just to avoid a warning
    if ('q' == key || 'Q' == key || 27 == key){
        delete colorim;
        exit(0);
    }else if(']' == key){
        pti ++;
        if(pti >=numofcuts){
            pti = numofcuts -1;
        }
    }else if('[' == key){
        pti --;
        if(pti < 0){
            pti = 0;
        }
    }
    
    printf("Zplane: %d\n", pti);
    
    glutPostRedisplay();
}


void ReshapeFunc(int width, int height)
{
    glViewport(0, 0, width, height);
}

int main(int args, char * argv[]){
    if(args != 2){
        printf("DensViewer <densfile>\n");
        exit(1);
    }
    

    float boxsize;
    float startz;
    float dz;
    int head[59];
    float * im;

    
    fstream file;
    file.open(argv[1], ios::in | ios::binary);
    file.read((char *) &gridsize, sizeof(int));
    file.read((char *) &numofcuts, sizeof(int));
    file.read((char *) &boxsize, sizeof(float));
    file.read((char *) &startz, sizeof(float));
    file.read((char *) &dz, sizeof(float));
    file.read((char *) head, sizeof(int) * 59);
    
    if(numofcuts <= 0){
        numofcuts = gridsize;
    }
    
    printf("Gridsize = %d NumofCuts = %d \n", gridsize, numofcuts);
    
    int pixels;
    

    pixels = gridsize * gridsize * numofcuts;

    
    
    im = new float[pixels];
    colorim = new float[pixels * 3];
    file.read((char *) im, sizeof(float) * pixels);
    
    file.close();
    
    //get color image
    printf("Calculating the color picture ...\n");
    getcolorImge(im, colorim, pixels);
    delete im;
    
    printf("Rendering...\n");
    //rendering...
    //initiate openGL
    glutInit(&args, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);
    glutInitWindowSize(gridsize, gridsize);
    glutCreateWindow("Dark Matter Density rendering!");


	#ifndef __APPLE__
    glewExperimental = GL_TRUE;
    glewInit();
	#endif
    
    
    glGenTextures(1, &(textureIni));
    glBindTexture(GL_TEXTURE_2D, textureIni);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, gridsize, gridsize, 0,
                 GL_RGB, GL_FLOAT, colorim + 0 * gridsize * gridsize);
    // set its parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    
    
    glutDisplayFunc(&rendsenc);
    glutReshapeFunc(&ReshapeFunc);
    glutKeyboardFunc(&KeyboardFunc);
    glutMainLoop();

    return 0;
    
}