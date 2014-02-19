#include <cstdlib>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <iostream>
#include <cmath>

#include "tetrahedron.h"
#include "dtetrastream.h"
#include "triconverter.h"
#include "triheader.h"
#include "trifile_util.h"
#include "processbar.h"


using namespace std;

string inputbase = "";
string outputbase = "";
int parttype = 1;
int imageSize = 512;
bool isRedShiftDist = false;
Point redshiftAxis;
int typeCode = 0x00;

/*
const int VELX =  0x01;
const int VELY =  0x02;
const int VELZ =  0x04;
const int POS  =  0x08;
const int DENS =  0x10;
 */

bool isPosition_ = true;
bool isDensity_ = true;
bool isVelX_ = false;
bool isVelY_ = false;
bool isVelZ_ = false;

void printUsage(string pname){
    printf("Usage: %s\n %s\n %s\n %s\n %s\n %s\n"
            " %s\n %s\n %s\n %s\n %s\n",
           pname.c_str(),
           "-df <Data Basename>",
           "-of <Output Basename>",
           "-imsize <imagesize>",
           "-pos    output position data",
           "-dens   output density data",
           "-velx   (TODO) output density weighted velocity",
           "-vely   (TODO) output density weighted velocity",
           "-velz   (TODO) output density weighted velocity",
           "-parttype  <particle type>, default: 1",
           "-redshift <x> <y> <z>, the reshift shift distortion axis" 
           );
}



void savefile(DtetraStream &streamer){
    if(isRedShiftDist){
        streamer.setRedShitDistortion(redshiftAxis);
    }
    
    
    TriConverter triangleConverter(imageSize,
                                   streamer.getHeader().boxSize);
    
    TriHeader theader;
    theader.ImageSize = imageSize;
    theader.boxSize = streamer.getHeader().boxSize;
    theader.startZ = 0;
    theader.dz = streamer.getHeader().boxSize / (double) imageSize;
    theader.numOfZPlanes = imageSize;
    
    TrifileWriter twriter(theader);
    twriter.open(outputbase);
    if(!twriter.good()){
        printf("Output error!\n");
        exit(1);
    }
    
    //triangleConverter.setOutput(typeCode);
    
    
    //int datagridsize = streamer.getHeader().gridsize;
     
    //int numTetras = 0;

    uint64_t tetra_count = 0;
    

    
    int count_ind = 0;
    int numfiles = streamer.getHeader().totalfiles;
    IndTetrahedron indtetra;
    
    
    //int *currentTriIdPlane = new int[imagesize];
    //memset(currentTriIdPlane, 0, sizeof(int) * imagesize);
    
    ProcessBar bar(numfiles * 100, 0);
    bar.start();
    for(int l = 0; l < numfiles; l++){
        streamer.loadBlock(l);
        int numindtetra = streamer.getNumTetras();
        IndTetrahedronManager & im = streamer.getCurrentIndtetraManeger();
        for(int i = 0; i < numindtetra; i ++){
            streamer.getIndTetra(indtetra, i);
            
            bar.setvalue(l * 100 + i * 100 / numindtetra);
            
            int nt = im.getNumPeriodical(indtetra);
            Tetrahedron * ts = im.getPeroidTetras(indtetra);
            count_ind ++;
            for(int k = 0; k < nt; k++){
                triangleConverter.process(ts[k]);
                
                //printf("Is Max: %d\n", triangleConverter.isReachMax());
                if(triangleConverter.isReachMax()){
                    //output
                    int *f_inds = new int[triangleConverter.getTotalTriangles()];
                    int *planetris = triangleConverter.getNumTrisInPlanes();
                    vector<int> &trianglePlaneIds_ = triangleConverter.getTrianglePlaneIds();
                    vector<float> &vertexData_ = triangleConverter.getVertex();
                    vector<float> &densityData_ = triangleConverter.getDensity();
                    
                    twriter.write(planetris, trianglePlaneIds_, vertexData_, densityData_);
                    //currentTriIdPlane[0] = 0;
                    
                    triangleConverter.reset();
                    delete[] f_inds;
                }
                
                
                tetra_count ++;
            }
            
        }
    }
    
    
    //output the final triangles
    if(triangleConverter.getTotalTriangles() > 0){
        int *f_inds = new int[triangleConverter.getTotalTriangles()];
        int *planetris = triangleConverter.getNumTrisInPlanes();
        vector<int> trianglePlaneIds_ = triangleConverter.getTrianglePlaneIds();
        vector<float> vertexData_ = triangleConverter.getVertex();
        vector<float> densityData_ = triangleConverter.getDensity();
        
        twriter.write(planetris, trianglePlaneIds_, vertexData_, densityData_);
        triangleConverter.reset();
        delete[] f_inds;
    }
    
    //numTetras = 0;
    bar.end();
    printf("Finished.\nIn total %lld tetrahedrons output.\n", tetra_count);
}


int main(int argv, char * args[]){

    
    int k = 1;
    if(argv == 1){
        printUsage(args[0]);
        exit(1);
    }else{
        while(k < argv){
            stringstream ss;
            if(strcmp(args[k], "-df") == 0){
                ss << args[k + 1];
                ss >> inputbase;
            }else if(strcmp(args[k], "-of") == 0){
                outputbase = args[k+1];
            }else if(strcmp(args[k], "-parttype") == 0){
                ss << args[k + 1];
                ss >> parttype;
            }else if(strcmp(args[k], "-imsize") == 0){
                ss << args[k + 1];
                ss >> imageSize;
            }else if(strcmp(args[k], "-pos") == 0){
                k --;
                typeCode = typeCode | POS;
            }else if(strcmp(args[k], "-dens") == 0){
                typeCode = typeCode | DENS;
                k --;
            }else if(strcmp(args[k], "-velx") == 0){
                typeCode = typeCode | VELX;
                k --;
            }else if(strcmp(args[k], "-vely") == 0){
                typeCode = typeCode | VELY;
                k --;
            }else if(strcmp(args[k], "-velz") == 0){
                typeCode = typeCode | VELZ;
                k --;
            }else if(strcmp(args[k], "-redshift") == 0){
                float r_x, r_y, r_z;
                stringstream s0;
                s0 << args[k + 1];
                s0 >> r_x;
                k++;
                stringstream s1;
                s1 << args[k+1];
                s1 >> r_y;
                k++;
                ss << args[k + 1];
                ss >> r_z;
                isRedShiftDist = true;
                float r = sqrt(r_x * r_x + r_y * r_y + r_z * r_z);
                r_x /= r;
                r_y /= r;
                r_z /= r;
                redshiftAxis.x = r_x;
                redshiftAxis.y = r_y;
                redshiftAxis.z = r_z;
            }else{
                printUsage(args[0]);
                exit(1);
            }
            k += 2;
        }

        if(inputbase == ""){
            printUsage(args[0]);
            exit(1);
        }
    }
    
    DtetraStream streamer(inputbase);

    savefile(streamer);

}
