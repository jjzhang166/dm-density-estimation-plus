#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <cmath>

#include "tetrahedron.h"
#include "indtetrastream.h"
#include "tfileheader.h"

#define BUFFERSIZE 65536
#define H0 100.0

using namespace std;

string prefix = "";
string base_name = "";
string singlefilename = "";
string outputfile = "";
int parttype = 1;
int numoffiles = 0;
int datagridsize = -1;
int inputmemgrid = -1;
Tetrahedron tetrabuffer[BUFFERSIZE + 20];

bool isRedShiftDist = false;  
Point redshiftAxis; //redshit distortion axis

void printUsage(string pname){
    printf("Usage: %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n",
           pname.c_str(),
           "-df <single_gadgetfile name>",
           "-mf <prefix> <basename> <numoffiles>",
           "-of <output t-file name>",
           "-parttype <particletype>, default: -1",
           "-dgridsize <data gridsize>, default: -1",
           "-tgrid <grid in memory for tetra>, default: -1",
           "-redshift <x> <y> <z>, the reshift shift distortion axis" 
           );
}



void getRedshiftDistoredPoint(Point & target,
                              Point & velocity,
                              Point & distortAxis,
                              float redshift
                              ){
    //printf("%f %f %f %f %f %f\n", target.x, target.y, target.z, velocity.x, velocity.y, velocity.z);
    
    float a = 1.0 / (1.0 + redshift);
    
    Point displacement = distortAxis
    * velocity.dot(distortAxis)
    * sqrt(a) * (1.0 / H0) * 1000; //to kpc/h

    
    target = target + displacement;
    
    //printf("%f %f %f\n", displacement.x, displacement.y, displacement.z);
}

void getRedshiftDistoredTetra(Tetrahedron &target,
                              Point & distortAxis,
                              float redshift){
    getRedshiftDistoredPoint(target.v1, target.velocity1, distortAxis, redshift);
    getRedshiftDistoredPoint(target.v2, target.velocity2, distortAxis, redshift);
    getRedshiftDistoredPoint(target.v3, target.velocity3, distortAxis, redshift);
    getRedshiftDistoredPoint(target.v4, target.velocity4, distortAxis, redshift);
    target.computeVolume();
}


void savefile(TetraStreamer &streamer){
    if(isRedShiftDist){
        streamer.setRedshiftDistort(redshiftAxis);
    }
    
    if(datagridsize == -1){
        datagridsize = (int)ceil(pow(streamer.getIndTetraStream()->getHeader().npartTotal[parttype], 1.0 / 3.0));
    }
    
    if(inputmemgrid == -1){
		inputmemgrid = datagridsize;
	}
   
     
    
    TFileHeader header;
    header.numOfTetrahedrons = 0l;
    header.boxSize = streamer.getIndTetraStream()->getHeader().BoxSize;
    int numTetras = 0;
   
    
    fstream outputstream(outputfile.c_str(), ios::out | ios::binary);
    if(!outputstream.good()){
        printf("Output file incorrect!!\n");
        exit(1);
    }
    
    uint64_t tetra_count = 0;
    uint64_t tcount = datagridsize * datagridsize * datagridsize / 10 * 6;
    if(tcount == 0){
        tcount = 1;
    }

    //printf("%d %d \n", tcount, datagridsize);
    streamer.reset();
    while(streamer.hasNext()){
        int nums;
        
        Tetrahedron * tetras;
        tetras = streamer.getNext(nums);

        for(int i= 0; i < nums; i++){
            //float boxsize =  streamer.getIndTetraStream()->getHeader().BoxSize;
            
            /*if(isRedShiftDist){
                
                Tetrahedron newtetra = tetras[i];
                getRedshiftDistoredTetra(newtetra, redshiftAxis,
                                         streamer.getIndTetraStream()
                                         ->getHeader().redshift);
                tetrabuffer[numTetras] = newtetra;
                
                if(tetras[i].maxx() > boxsize && tetras[i].minx() < boxsize && newtetra.minx() > boxsize){
                    tetra_count ++;
                    numTetras ++;
                    tetrabuffer[numTetras] = newtetra - boxsize;
                }
                
                if(tetras[i].maxx() < 0){
                    tetrabuffer[numTetras].v1.x += boxsize;
                    tetrabuffer[numTetras].v2.x += boxsize;
                    tetrabuffer[numTetras].v3.x += boxsize;
                    tetrabuffer[numTetras].v4.x += boxsize;
                }else if(tetras[i].minx() > boxsize){
                    tetrabuffer[numTetras].v1.x -= boxsize;
                    tetrabuffer[numTetras].v2.x -= boxsize;
                    tetrabuffer[numTetras].v3.x -= boxsize;
                    tetrabuffer[numTetras].v4.x -= boxsize;
                }
                
                if(tetras[i].maxy() < 0){
                    tetrabuffer[numTetras].v1.y += boxsize;
                    tetrabuffer[numTetras].v2.y += boxsize;
                    tetrabuffer[numTetras].v3.y += boxsize;
                    tetrabuffer[numTetras].v4.y += boxsize;
                }else if(tetras[i].miny() > boxsize){
                    tetrabuffer[numTetras].v1.y -= boxsize;
                    tetrabuffer[numTetras].v2.y -= boxsize;
                    tetrabuffer[numTetras].v3.y -= boxsize;
                    tetrabuffer[numTetras].v4.y -= boxsize;
                }
                
                if(tetras[i].maxz() < 0){
                    tetrabuffer[numTetras].v1.z += boxsize;
                    tetrabuffer[numTetras].v2.z += boxsize;
                    tetrabuffer[numTetras].v3.z += boxsize;
                    tetrabuffer[numTetras].v4.z += boxsize;
                }else if(tetras[i].minz() > boxsize){
                    tetrabuffer[numTetras].v1.z -= boxsize;
                    tetrabuffer[numTetras].v2.z -= boxsize;
                    tetrabuffer[numTetras].v3.z -= boxsize;
                    tetrabuffer[numTetras].v4.z -= boxsize;
                }
                
            }else*/{
                tetrabuffer[numTetras] = tetras[i];
            }
            
            
            
            numTetras ++;
            //printf("ok %d\n", tetra_count);
            if(numTetras >= BUFFERSIZE){
                //write to file
                outputstream.write((char *) tetrabuffer,
                                   numTetras * sizeof(Tetrahedron));
                numTetras = 0;
            }
            
            if((tetra_count %  tcount )== 0){
                printf(">");
                cout.flush();
            }
            tetra_count ++;
        }
        
        //printf("ok1 %d\n", tetra_count);
    }
    outputstream.write((char *) tetrabuffer,
                       numTetras * sizeof(Tetrahedron));
    numTetras = 0;
    header.numOfTetrahedrons = tetra_count;
    outputstream.seekg(0, outputstream.beg);
    outputstream.write((char *) &header, sizeof(TFileHeader));
    outputstream.close();
    printf("\nFinished. In total %ld tetrahedrons output.\n", (long) tetra_count);
}


int main(int argv, char * args[]){

    
    int k = 1;
    if(argv == 1){
        printUsage(args[0]);
        exit(1);
    }else{
        while(k < argv){
            stringstream ss;
            //printf("%s\n", args[k]);
            if(strcmp(args[k], "-df") == 0){
                ss << args[k + 1];
                ss >> singlefilename;
            }else if(strcmp(args[k], "-mf") == 0){
                prefix = args[k + 1];
                k++;
                base_name = args[k + 1];
                k++;
                ss << args[k + 1];
                ss >> numoffiles;
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
            }else if(strcmp(args[k], "-of") == 0){
                ss << args[k + 1];
                ss >> outputfile;
            }else if(strcmp(args[k], "-dgridsize") == 0){
                ss << args[k + 1];
                ss >> datagridsize;
            }else if(strcmp(args[k], "-parttype") == 0){
                ss << args[k + 1];
                ss >> parttype;
            }else if(strcmp(args[k], "-tgrid") == 0){
                ss << args[k + 1];
                ss >> inputmemgrid;
            }else{
                printUsage(args[0]);
                exit(1);
            }
            k += 2;
        }
        //printf("%d\n", numoffiles);

        if(singlefilename == "" && numoffiles == 0){
            printUsage(args[0]);
            exit(1);
        }
    }
    
    
    

    
    if(numoffiles != 0){
       //printf("ok1\n"); 
       TetraStreamer streamer(prefix,
                               base_name,
                               numoffiles,
                               inputmemgrid,
                               parttype,
                               datagridsize,
                               true,
                               true,
                               true,
                               true,
                               false);
       //printf("ok2\n"); 
       savefile(streamer);
        
    }else{
        TetraStreamer streamer(singlefilename,
                               inputmemgrid,
                               parttype,
                               datagridsize,
                               true,
                               true,
                               true,
                               true,
                               false);
        savefile(streamer);
        
    }
}
