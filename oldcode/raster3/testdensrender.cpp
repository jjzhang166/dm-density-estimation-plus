#include <cstdio>
#include <iostream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
//#include "gettimeofday_win.h"
#else
#include <unistd.h>
#include <sys/time.h>
#endif

#ifdef __APPLE__
#include <GLUT/glut.h> // darwin uses glut.h rather than GL/glut.h
#else
#include <GL/glew.h>
#include <GL/glut.h>
#endif


//#include "grid.h"
#include "tetrahedron.h"
#include "indtetrastream.h"
#include "isoplane.h"
#include "render.h"
#include "denrender.h"


#define NUMP 512


float * colorimage;

GLuint textureIni;


namespace main_space{
    
    int inputmemgrid = 16;						//the input memory grid size
    string filename =  "E:\\multires_150";//"I:\\data\\MIP-00-00-00-run_050";		//the input data filename "E:\\multires_150";//
    string gridfilename = "I:\\sandbox\\tetrahedron.grid";	//output filename
    string velofilename = "I:\\sandbox\\tetrahedron.vgrid";	//velocity output filename
    bool isoutputres = false;
    bool isVerbose = false;
    bool isInOrder = false;
    bool isVelocity = false;								//calculate velocity field?
    
    //load all the particles into memory?
    bool isHighMem = true;
    //return the all particle pointer?
    bool isAllData = false;
    
    //if use -1, then use the particle gridsize as the gridsize
    //otherwise use the user setting
    int datagridsize = -1;
    //the particle type in the gadget file
    int parttype = 1;
    
    bool isSetBox = false;                       //set up a box for the grids
    Point setStartPoint;
    double boxsize = 32000.0;
    int imagesize = 512;
    double showz = 26300;                        //the showing z-direction
    
    void printUsage(string pname){
        fprintf(stderr, "Usage: %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n"
                , pname.c_str()
                , "[-imsize <imagesize>]"
                , "[-df <datafilename>]"
                , "[-of <gridfilename>]"
                , "[-vfile <velocityfieldfilename> only applied when use -vel]"
                , "[-t <numbers of tetra in memory>]"
                , "[-o] to output result in texts"
                , "[-order] if the data is in order"
                , "[-v] to show verbose"
                , "[-vel] if calculate velocity field"
                , "[-dgridsize] default: -1 to use the npart^(1/3) as gridsize"
                , "[-parttype] default: 1. Use 0-NTYPE data in the gadgetfile"
                , "[-lowmem] use low memory mode (don't load all part in mem)"
                , "[-nalldata] only usable in highmem mode"
                , "[-box <x0> <y0> <z0> <boxsize>] setup the start point, and the boxsize. The box should be inside the data's box, otherwise some unpredictable side effects will comes out"
                , "[-showz <z-coor of the cut plane>]"
                );
    }
    
    void readParameters(int argv, char * args[]){
        int k = 1;
        if(argv == 1){
            return;
        }else{
            while(k < argv){
                stringstream ss;
                if(strcmp(args[k], "-imsize") == 0){
                    ss << args[k + 1];
                    ss >> imagesize;
                }else if(strcmp(args[k], "-df") == 0){
                    ss << args[k + 1];
                    ss >> filename;
                }else if(strcmp(args[k], "-of") == 0){
                    ss << args[k + 1];
                    ss >> gridfilename;
                }else if(strcmp(args[k], "-vfile") == 0){
                    ss << args[k + 1];
                    ss >> velofilename;
                }else if(strcmp(args[k], "-t") == 0){
                    ss << args[k + 1];
                    ss >> inputmemgrid;
                }else if(strcmp(args[k], "-o") == 0){
                    isoutputres = true;
                    k = k -1;
                }else if(strcmp(args[k], "-v") == 0){
                    isVerbose = true;
                    k = k -1;
                }else if(strcmp(args[k], "-order") == 0){
                    isInOrder = true;
                    k = k -1;
                }else if(strcmp(args[k], "-vel") == 0){
                    isVelocity = true;
                    k = k -1;
                }else if(strcmp(args[k], "-dgridsize") == 0){
                    ss << args[k + 1];
                    ss >> datagridsize;
                }else if(strcmp(args[k], "-parttype") == 0){
                    ss << args[k + 1];
                    ss >> parttype;
                }else if(strcmp(args[k], "-lowmem") == 0){
                    isHighMem = false;
                    k = k -1;
                }else if(strcmp(args[k], "-alldata") == 0){
                    isAllData = true;
                    k = k -1;
                }else if(strcmp(args[k], "-showz") == 0){
                    ss << args[k + 1];
                    ss >> showz;
                }else if(strcmp(args[k], "-box") == 0){
                    isSetBox = true;
                    k++;
                    ss << args[k] << " ";
                    ss << args[k + 1] << " ";
                    ss << args[k + 2] << " ";
                    ss << args[k + 3] << " ";
                    ss >> setStartPoint.x;
                    ss >> setStartPoint.y;
                    ss >> setStartPoint.z;
                    ss >> boxsize;
                    k += 2;
                }else{
                    printUsage(args[0]);
                    exit(1);
                }
                k += 2;
            }
        }
    }
    
}





using namespace main_space;


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
    for(int i = 0; i < imagesize * imagesize; i++){
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
    for(int i = 0; i < imagesize * imagesize; i++){
        float v = (log(value[i]) - log(min)) / x;
        //printf("%f\n",v);
        getJetColor(v, r, g, b);
        colorimg[3 * i] = r;
        colorimg[3 * i + 1] = g;
        colorimg[3 * i + 2] = b;
    }
}






int pti = 0;


void rendsenc(){
    //glClientActiveTexture(GL_TEXTURE0);
	glViewport(0,0,imagesize, imagesize);
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
    printf("Texture: %d\n", pti);
    //glBindTexture(GL_TEXTURE_2D, textureIni[pti]);
    
    glBindTexture(GL_TEXTURE_2D, textureIni);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imagesize, imagesize, 0,
                 GL_RGB, GL_FLOAT, colorimage + pti * 3 * imagesize * imagesize);
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
    }else if(']' == key){
        pti ++;
        if(pti >=NUMP){
            pti = NUMP -1;
        }
    }else if('[' == key){
        pti --;
        if(pti < 0){
            pti = 0;
        }
    }
    
    printf(": %d\n", pti);
    
    glutPostRedisplay();
}





int main(int argv, char * args[]){
	//double io_t = 0, calc_t = 0, total_t = 0;
	//timeval timediff;
	//double t1, t2, t0 = 0;
	
	//gettimeofday(&timediff, NULL);
	//t0 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
    
	readParameters(argv, args);
	printf("\n=========================DENSITY ESTIMATION==========================\n");
	printf("*****************************PARAMETERES*****************************\n");
    printf("Render Image Size       = %d\n", imagesize);
	printf("Data File               = %s\n", filename.c_str());
    if(datagridsize == -1){
        printf("DataGridsize            = [to be determined by data]\n");
    }else{
        printf("DataGridsize            = %d\n", datagridsize);
    }
    printf("Particle Type           = %d\n", parttype);
	printf("Grid File               = %s\n", gridfilename.c_str());
	printf("Tetra in Mem            = %d\n", inputmemgrid);
    
    if(isSetBox){
        printf("Box                    = %f %f %f %f\n",
               setStartPoint.x, setStartPoint.y, setStartPoint.z, boxsize);
    }
	if(isVelocity){
		printf("Vel File               = %s\n", velofilename.c_str());
	}
	if(isInOrder){
		printf("The data is already in right order for speed up...\n");
	}
    if(!isHighMem){
        printf("Low Memory mode: slower in reading file...\n");
    }else{
        printf("Block Memory Operation:\n");
        if(!isAllData){
            printf("    Use Memory Copy Mode -- but may be faster without regenerating the tetras...\n");
        }else{
            printf("    Without Memory Copying Mode -- but may be slower in regenerating tetras...\n");
        }
        
    }
    
    printf("*********************************************************************\n");
    
    

    
    colorimage = new float[imagesize * imagesize * NUMP * 3];
    
    //test
    TetraStreamer streamer(filename,
                           inputmemgrid,
                           parttype,
                           datagridsize,
                           isHighMem,
                           isAllData,
                           isVelocity,
                           true,
                           isInOrder);
    boxsize = streamer.getIndTetraStream()->getHeader().BoxSize;
    
    vector<RenderType> renderTypes;
    renderTypes.push_back(DENSITY);
    DenRender render(imagesize, boxsize,
                     0, boxsize / NUMP, NUMP,
                     renderTypes);
    
    printf("Boxsize: %f\n", boxsize);
    
    int count = 0;
    
    //render
    
    while(streamer.hasNext()){
        int nums;
        Tetrahedron * tetras;
        tetras = streamer.getNext(nums);
        for(int i= 0; i < nums; i++){
            //printf("Oz = %f %f %f %f\n",
            //      tetras[i].v1.z, tetras[i].v2.z, tetras[i].v3.z, tetras[i].v4.z);
            render.rend(tetras[i]);
        }
        count += nums;
    }
    render.finish();
    float * im = render.getResult();
    
    printf("Starting compute color image!\n");
    
    //get color image
    for(int i = 0; i < NUMP; i++){
        getcolorImge(im + imagesize * imagesize * i,
                     colorimage + imagesize * imagesize * 3 * i);
    }
    
    //for(int i = 0; i < NUMP; i++){
    glGenTextures(1, &(textureIni));
    glBindTexture(GL_TEXTURE_2D, textureIni);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imagesize, imagesize, 0,
                 GL_RGB, GL_FLOAT, colorimage + 0 * imagesize * imagesize);
    // set its parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    //}
  
    
    glutDisplayFunc(&rendsenc);
    glutReshapeFunc(&ReshapeFunc);
    glutKeyboardFunc(&KeyboardFunc);
    glutMainLoop();
    
    return 0;
}


