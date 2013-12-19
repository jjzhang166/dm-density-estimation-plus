#include "GadgetReader/gadgetreader.hpp"
#include "GadgetReader/gadgetheader.h"
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <stdint.h>
#include <cstdio>
#include <vector>
#include <fstream>
#include <sstream>
#include "io_utils.h"
#include "processbar.h"


using namespace std;
using namespace GadgetReader;


#define BUFFERSIZE 1024*1024

long long ** inds;
float ** pos;
float ** vel;
int * currentParticlesInBuffer;
int partPerBuffer;

string outputbase;
string inputfilename;
divide_header * header;


void pushBackParticle(vector<long long> & tempind, vector<float> & temppos,
                      vector<float> & tempvel, int ind, int j, int blockj){
    inds[blockj][currentParticlesInBuffer[blockj]] = ind;//tempind[j];//.push_back(tempind[j]);
    pos[blockj][currentParticlesInBuffer[blockj]]=temppos[j * 3 + 0];//.push_back(temppos[j * 3 + 0]);
    pos[blockj][currentParticlesInBuffer[blockj]]=temppos[j * 3 + 1];//.push_back(temppos[j * 3 + 1]);
    pos[blockj][currentParticlesInBuffer[blockj]]=temppos[j * 3 + 2];//.push_back(temppos[j * 3 + 2]);
    vel[blockj][currentParticlesInBuffer[blockj]]=tempvel[j * 3 + 0];//.push_back(tempvel[j * 3 + 0]);
    vel[blockj][currentParticlesInBuffer[blockj]]=tempvel[j * 3 + 1];//.push_back(tempvel[j * 3 + 1]);
    vel[blockj][currentParticlesInBuffer[blockj]]=tempvel[j * 3 + 2];//.push_back(tempvel[j * 3 + 2]);
    currentParticlesInBuffer[blockj] ++;
    if(currentParticlesInBuffer[blockj] >= partPerBuffer){
        writeToDividerFile(outputbase,
                           INDEXTYPE,
                           j,
                           ios::out | ios::binary | ios::app,
                           ((char *) inds[j]),//.data()),
                           sizeof(long long) * currentParticlesInBuffer[blockj]//inds[j].size()
                           );
        
        writeToDividerFile(outputbase,
                           VELTYPE,
                           j,
                           ios::out | ios::binary | ios::app,
                           ((char *) vel[j]),//.data()),
                           sizeof(float) * currentParticlesInBuffer[blockj]//vel[j].size()
                           );
        
        writeToDividerFile(outputbase,
                           POSTYPE,
                           j,
                           ios::out | ios::binary | ios::app,
                           ((char *) pos[j]),//.data()),
                           sizeof(float) * currentParticlesInBuffer[blockj]//pos[j].size()
                           );
        
        header[blockj].numparts += currentParticlesInBuffer[blockj];
        currentParticlesInBuffer[blockj] = 0;
    }
}



int main(int argv, char * args[]){
    if(argv != 5 && argv != 6){
        printf("%s Filename NumSlicePerBlock PartType OutputBase [GridSize]\n", args[0]);
        exit(1);
    }
    inputfilename = args[1];
    
    //printf("%d\n", sizeof(divide_header));
    
    int numslice = 0;
    int numzPerTrunk = atoi(args[2]);
    int parttype = atoi(args[3]);
    long long startID = 0;
    
    outputbase = args[4];
    printf("Input File: %s\n", inputfilename.c_str());
    printf("Numbers of slices per trunk %d\n", numzPerTrunk);
    printf("OutputBase: %s\n", outputbase.c_str());
    printf("Particle Type: %d\n", parttype);
    int gridsize = -1;
    if(argv == 6){
        gridsize = atoi(args[5]);
    }
    
    //dividing

    GSnap snap(inputfilename, false);
    if(!snap.IsBlock("POS ")){
        printf("No Position Block!\n");
        exit(1);
    }
    if(!snap.IsBlock("VEL ")){
        printf("No Velocity Block!\n");
        exit(1);
    }
    

    
    int64_t nparts = snap.GetNpart(parttype);
    if(gridsize == -1){
        gridsize = ceil(pow(nparts, 1.0/3.0));
    }


    
    
    printf("Number of Particles: %lld\n", nparts);
    printf("Grid Size: %d\n", gridsize);
    printf("Dividing ...\n");
    
    if(numzPerTrunk > gridsize){
        printf("WARNING: NumzPerTrunk larger than Gridsize\n");
        numzPerTrunk = gridsize;
    }
    
    if(numzPerTrunk <= 0){
        numzPerTrunk = 1;
    }
    
    numslice = gridsize / numzPerTrunk;
    if(numslice == 0){
        numslice = 1;
    }


    if((gridsize % numzPerTrunk) != 0){
        numslice += 1;
        //header.totalfiles += 1;
    }
    int totalfiles = numslice;
    
    header = new divide_header[totalfiles];
    
    //create file
    for(int i = 0; i < totalfiles; i++){
        header[i].totalfiles = totalfiles;
        header[i].gridsize = gridsize;
        header[i].numofZgrids = numzPerTrunk + 1;
        header[i].boxSize = snap.GetHeader().BoxSize;
        header[i].filenumber = i;
        header[i].startind = gridsize * gridsize * i * numzPerTrunk;
        header[i].numparts = 0;
        header[i].redshift = snap.GetHeader().redshift;
        if(numzPerTrunk >= gridsize - i * numzPerTrunk){
            header[i].numofZgrids = gridsize - i * numzPerTrunk + 1;
        }
        
        writeToDividerFile(outputbase,
                    INDEXTYPE,
                    i,
                    ios::out | ios::binary,
                    ((char *) &(header[i])),
                    sizeof(header[i])
                    );
        
        writeToDividerFile(outputbase,
                    VELTYPE,
                    i,
                    ios::out | ios::binary,
                    (char *) &(header[i]),
                    sizeof(header[i])
                    );
        
        writeToDividerFile(outputbase,
                    POSTYPE,
                    i,
                    ios::out | ios::binary,
                    (char *) &(header[i]),
                    sizeof(header[i])
                    );
        
        //printf();
    }
    
    //vector<long long> * inds = new vector<long long>[totalfiles];
    //vector<float> * pos = new vector<float>[totalfiles];
    //vector<float> * vel = new vector<float>[totalfiles];
    inds = new long long* [totalfiles];
    pos = new float* [totalfiles];
    vel = new float* [totalfiles];
    currentParticlesInBuffer = new int[totalfiles];
    partPerBuffer = BUFFERSIZE / totalfiles + 1;
    
    for(int  i = 0; i < totalfiles; i++){
        currentParticlesInBuffer[i] = 0;
        inds[i] = new long long[partPerBuffer];
        pos[i] = new float[partPerBuffer];
        vel[i] = new float[partPerBuffer];
    }

    int64_t numpartsToRead = numzPerTrunk * gridsize * gridsize;
    
    
    //read particles from file, divide them
    unsigned int ignorecode = ~(1 << parttype);
    
    startID = (long long)gridsize * (long long)gridsize * (long long)gridsize * (long long)1024;
    for(int i = 0; i < nparts; i+= numpartsToRead){
        vector<long long> tempind = snap.GetBlockInt("ID  ", numpartsToRead, i, ignorecode);
        long long tempmin = *(std::min_element(tempind.begin(), tempind.end()));
        //printf("%ld, startID, %ld\n", startID, tempmin);
        if(tempmin < startID){
            startID = tempmin;
        }
    }
    //printf("%d, startID\n", startID);
    int partCounts = 0;

    
    ProcessBar bar(totalfiles, 0);
    bar.start();
    for(int i = 0; i < totalfiles; i++){
        //for(int j = 0; j < totalfiles; j++){
            //inds[j].clear();
            //pos[j].clear();
            //vel[j].clear();
        //    currentParticlesInBuffer[j] = 0;
        //}
        memset (currentParticlesInBuffer, 0, sizeof(int) * totalfiles);
        
        bar.setvalue(i);
        
        
        int startind = header[i].startind;
        vector<float> temppos = snap.GetBlock("POS ", numpartsToRead, startind, ignorecode);
        

        //int possize = snap.GetBlock("POS ", temppos, numpartsToRead, startind, ignorecode);
        vector<float> tempvel = snap.GetBlock("VEL ", numpartsToRead, startind, ignorecode);
        

        // int velsize = snap.GetBlock("VEL ", tempvel, numpartsToRead, startind, ignorecode);
        vector<long long> tempind = snap.GetBlockInt("ID  ", numpartsToRead, startind, ignorecode);
        
        
        int indsize = tempind.size();
        //int indsize = snap.GetBlock("ID  ", tempind, numpartsToRead, startind, ignorecode);

        //printf("SIZE: %d, %d, %d, %u\n", indsize, numpartsToRead, startind, ignorecode);
        for(int j = 0; j < indsize; j++){
            partCounts ++;
            tempind[j] -= startID;
            long long z = tempind[j] / (gridsize * gridsize);
            long long blockj = z / numzPerTrunk;
            //printf("ok %d %d %ld, %d, %d\n", indsize, j, tempind[j], z, numzPerTrunk);
            //printf("ok %ld %d %d %d\n", tempind[j], j, blockj, startID);
            
            
            /*inds[currentParticlesInBuffer[blockj]] = tempind[j];//.push_back(tempind[j]);
            pos[currentParticlesInBuffer[blockj]]=temppos[j * 3 + 0];//.push_back(temppos[j * 3 + 0]);
            pos[currentParticlesInBuffer[blockj]]=temppos[j * 3 + 1];//.push_back(temppos[j * 3 + 1]);
            pos[currentParticlesInBuffer[blockj]]=temppos[j * 3 + 2];//.push_back(temppos[j * 3 + 2]);
            vel[currentParticlesInBuffer[blockj]]=tempvel[j * 3 + 0];//.push_back(tempvel[j * 3 + 0]);
            vel[currentParticlesInBuffer[blockj]]=tempvel[j * 3 + 1];//.push_back(tempvel[j * 3 + 1]);
            vel[currentParticlesInBuffer[blockj]]=tempvel[j * 3 + 2];//.push_back(tempvel[j * 3 + 2]);
            currentParticlesInBuffer[blockj] ++;
            if(currentParticlesInBuffer[blockj] >= partPerBuffer){
                writeToDividerFile(outputbase,
                                   INDEXTYPE,
                                   j,
                                   ios::out | ios::binary | ios::app,
                                   ((char *) inds[j]),//.data()),
                                   sizeof(long long) * currentParticlesInBuffer[blockj]//inds[j].size()
                                   );
                
                writeToDividerFile(outputbase,
                                   VELTYPE,
                                   j,
                                   ios::out | ios::binary | ios::app,
                                   ((char *) vel[j]),//.data()),
                                   sizeof(float) * currentParticlesInBuffer[blockj]//vel[j].size()
                                   );
                
                writeToDividerFile(outputbase,
                                   POSTYPE,
                                   j,
                                   ios::out | ios::binary | ios::app,
                                   ((char *) pos[j]),//.data()),
                                   sizeof(float) * currentParticlesInBuffer[blockj]//pos[j].size()
                                   );
                                   
                currentParticlesInBuffer[blockj] = 0;
            }*/
            pushBackParticle(tempind, temppos, tempvel, tempind[j], j, blockj);
            
            
            
            if(z % numzPerTrunk == 0){
                blockj = blockj - 1;
                
                int ind1 = tempind[j];
                if(blockj < 0){
                    blockj = totalfiles - 1;
                    ind1 = tempind[j] + gridsize * gridsize * gridsize;
                }
                /*inds[blockj].push_back(ind1);
                pos[blockj].push_back(temppos[j * 3 + 0]);
                pos[blockj].push_back(temppos[j * 3 + 1]);
                pos[blockj].push_back(temppos[j * 3 + 2]);
                vel[blockj].push_back(tempvel[j * 3 + 0]);
                vel[blockj].push_back(tempvel[j * 3 + 1]);
                vel[blockj].push_back(tempvel[j * 3 + 2]);*/
                pushBackParticle(tempind, temppos, tempvel, ind1, j, blockj);
            }
        }

        //write the data to file
        for(int j = 0; j < totalfiles; j++){
            //printf("ok %d %d\n", i, j);
            header[j].numparts += currentParticlesInBuffer[j];//inds[j].size();
            
            writeToDividerFile(outputbase,
                        INDEXTYPE,
                        j,
                        ios::out | ios::binary | ios::app,
                        ((char *) inds[j]),//.data()),
                        sizeof(long long) * currentParticlesInBuffer[j]//* inds[j].size()
                        );
            
            writeToDividerFile(outputbase,
                        VELTYPE,
                        j,
                        ios::out | ios::binary | ios::app,
                        ((char *) vel[j]),//.data()),
                        sizeof(float) * currentParticlesInBuffer[j]//* vel[j].size()
                        );
            
            writeToDividerFile(outputbase,
                        POSTYPE,
                        j,
                        ios::out | ios::binary | ios::app,
                        ((char *) pos[j]),//.data()),
                        sizeof(float) * currentParticlesInBuffer[j]//* pos[j].size()
                        );
            currentParticlesInBuffer[j] = 0;
        }
    }
    
    //finish:
    for(int i = 0; i < totalfiles; i++){
        writeToDividerFile(outputbase,
                    INDEXTYPE,
                    i,
                    ios::out | ios::binary | ios::in,
                    ((char *) &(header[i])),
                    sizeof(header[i]),
                    true
                    );
        
        writeToDividerFile(outputbase,
                    VELTYPE,
                    i,
                    ios::out | ios::binary | ios::in,
                    ((char *) &(header[i])),
                    sizeof(header[i]),
                    true
                    );
        
        writeToDividerFile(outputbase,
                    POSTYPE,
                    i,
                    ios::out | ios::binary | ios::in,
                    ((char *) &(header[i])),
                    sizeof(header[i]),
                    true
                    );
    }
    for(int  i = 0; i < totalfiles; i++){
        //currentParticlesInBuffer[i] = 0;
        delete[] inds[i];
        delete[] pos[i];
        delete[] vel[i];
    }
    delete[] currentParticlesInBuffer;
    delete[] header;
    delete[] inds;
    delete[] pos;
    delete[] vel;
    bar.end();
    printf("Finished: %d particle processed.\n", partCounts);
}
