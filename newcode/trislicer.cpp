/******************************************************
 * Slice the data block file into slices of triangles.
 * Author: Lin F. Yang
 * Date: Feb 2014
 ******************************************************/

#include <cstdlib>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cinttypes>

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


bool isVelocity = false;

void printUsage(string pname){
    printf("Usage: %s\n %s\n %s\n %s\n %s\n %s\n %s\n",
           pname.c_str(),
           "-df <Data Basename>",
           "-of <Output Basename>",
           "-imsize <imagesize>",
           "-vel output velocity",
           "-parttype  <particle type>, default: 1",
           "-redshift <x> <y> <z>, the reshift shift distortion axis" 
           );
}


/**Convert triangles by reading tetrahedra data from the streamer */
void savefile(DtetraStream &streamer){
    /* Set the redshift distortion */
    if(isRedShiftDist){
        streamer.setRedShitDistortion(redshiftAxis);
    }
    
    
    /* Setup the triangle converter */
    TriConverter triangleConverter(imageSize,
                                   streamer.getHeader().boxSize,
                                   1024*1024 * 8,
                                   isVelocity);
    
    

    /* Triangle file header */ 
    TriHeader theader;
    theader.ImageSize = imageSize;
    theader.boxSize = streamer.getHeader().boxSize;
    theader.startZ = 0;
    theader.dz = streamer.getHeader().boxSize / (double) imageSize;
    theader.numOfZPlanes = imageSize;
    
    /* Triangle file writer */
    TrifileWriter twriter(theader, isVelocity);
    twriter.open(outputbase);
    if(!twriter.good()){
        printf("Output error!\n");
        exit(1);
    }
    

    uint64_t tetra_count = 0;
    

     
    int count_ind = 0;
    int numfiles = streamer.getHeader().totalfiles;
    IndTetrahedron indtetra;
    
    
    /* Progress bar */ 
    ProcessBar bar(numfiles * 100, 0);
    bar.start();
    
    /* Start converting tetrahedra to triangles */
    for(int l = 0; l < numfiles; l++){
        streamer.loadBlock(l);
        int numindtetra = streamer.getNumTetras();

        /* Using the object of index tetrahedra */
        IndTetrahedronManager & im = streamer.getCurrentIndtetraManeger();
        for(int i = 0; i < numindtetra; i ++){

            
            /* Get the i-th index tetrahera by reference */
            streamer.getIndTetra(indtetra, i);


            /* Update the progress bar */
            bar.setvalue(l * 100 + i * 100 / numindtetra);
            
            /* Deal with the peoridical condition */
            int nt = im.getNumPeriodical(indtetra);
            Tetrahedron * ts = im.getPeroidTetras(indtetra);
            
            /* Counts the number of index tetrahedra */
            count_ind ++;
            
            for(int k = 0; k < nt; k++){

                /* Convert the tetrahedra to be a list of triangles*/
                /* These triangles are saving the memory buffer*/
                triangleConverter.process(ts[k]);
               
                /* If memory buffer is full, then need save all the triangles
                 * into the hard drive, and clean the memory buffer.*/
                if(triangleConverter.isReachMax()){
                    

                    //int *f_inds = new int[triangleConverter.getTotalTriangles()];

                    /* Output the counts of triangles in each plane. */
                    int *planetris = triangleConverter.getNumTrisInPlanes();
                   
                    /* Each triangle's plane id */
                    vector<int> &trianglePlaneIds_ = triangleConverter.getTrianglePlaneIds();
                    
                    /* Vertex data of the triangle list */
                    vector<float> &vertexData_ = triangleConverter.getVertex();
                    
                    /* The density data of the triangle list */
                    vector<float> &densityData_ = triangleConverter.getDensity();
                    
                    /* These are the velocity field of these triangles */
                    vector<float> &velxData_ = triangleConverter.getVelocityX();
                    vector<float> &velyData_ = triangleConverter.getVelocityY();
                    vector<float> &velzData_ = triangleConverter.getVelocityZ();
                    
                    
                    /* write the data into memory */
                    if(!isVelocity){
                        twriter.write(planetris,
                                      &trianglePlaneIds_,
                                      &vertexData_,
                                      &densityData_);
                    }else{
                        twriter.write(planetris,
                                      &trianglePlaneIds_,
                                      &vertexData_,
                                      &densityData_,
                                      &velxData_,
                                      &velyData_,
                                      &velzData_);
                    }
                    
                    triangleConverter.reset();
                    //delete[] f_inds;
                }
                
                
                tetra_count ++;
            }
            
        }
    }
    
    
    /* Output the final portion triangles */
    if(triangleConverter.getTotalTriangles() > 0){
        //int *f_inds = new int[triangleConverter.getTotalTriangles()];
        int *planetris = triangleConverter.getNumTrisInPlanes();
        vector<int> trianglePlaneIds_ = triangleConverter.getTrianglePlaneIds();
        vector<float> vertexData_ = triangleConverter.getVertex();
        vector<float> densityData_ = triangleConverter.getDensity();
        
        vector<float> &velxData_ = triangleConverter.getVelocityX();
        vector<float> &velyData_ = triangleConverter.getVelocityY();
        vector<float> &velzData_ = triangleConverter.getVelocityZ();
        
        if(!isVelocity){
            twriter.write(planetris,
                          &trianglePlaneIds_,
                          &vertexData_,
                          &densityData_);
        }else{
            twriter.write(planetris,
                          &trianglePlaneIds_,
                          &vertexData_,
                          &densityData_,
                          &velxData_,
                          &velyData_,
                          &velzData_);
        }
        
        
        triangleConverter.reset();
        //delete[] f_inds;
    }
    
    
    /* End the progress bar */
    bar.end();
    printf("Finished.\nIn total %" PRIu64 " tetrahedrons output.\n", tetra_count);
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
            }else if(strcmp(args[k], "-vel") == 0){
                isVelocity = true;
                k -= 1;
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
    
    /* Data stream of tetrahedra */
    DtetraStream streamer(inputbase);

    savefile(streamer);

}
