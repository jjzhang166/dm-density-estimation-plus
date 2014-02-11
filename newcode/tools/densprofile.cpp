// Measure the radial bulk velocity profile
// Definition:
// For each shell
// v = \sum(m_i*(v_i \cdot \hat{r})) / \sum{m_i}

#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <cstring>

#include "gadgetreader.hpp"
#include "gadgetheader.h"

#ifndef PI
#define PI 3.141592653589793238462643383279
#endif


using namespace std;
using namespace GadgetReader;

void printUsage(const char * s){
    printf("Usage:\n"
           "%s\n"
           "-f <filename/basename>\n"
           "-nb <num of bins>\n"
           "-r <radius>\n"
           "-c <x>, <y>, <z>\n"
           "-t <particle type [0-6]>\n",
           s);
}


int main(int argc, const char **argv){
    int numbins;
    double radius;
    //double theta, phi;
    double * velbins;
    double * countbins;
    double * massbins;
    double * densbins;
    double dr, x, y, z;
    string filename = "";
    string maskfile = "";
    string baseName = "";
    //bool isMask = false;
    //bool isCore = false;
    //bool isMultFiles = false;
    int parttype = 1;
    
    int64_t bufferSize = 64*64*64;
    
    int m=1;
    while (m<argc)
    {
        string arg = argv[m];
        stringstream ss, ss1, ss2;
        if (arg == "-f") {
            filename = argv[m+1];
            m+=1;
        }else if (arg == "-nb") {
            ss << argv[m+1];
            ss >> numbins;
            m+=1;
        }else if (arg == "-r") {
            ss << argv[m+1];
            ss >> radius;
            m+=1;
        }else if (arg == "-c") {
            ss << argv[m+1];
            ss >> x;
            m++;
            
            ss1 << argv[m+1];
            ss1 >> y;
            m++;
            
            ss2 << argv[m+1];
            ss2 >> z;
            m++;
        }else if (arg == "-t") {
            ss << argv[m+1];
            ss >> parttype;
            m+=1;
        }else{
            printUsage(argv[0]);
            exit(1);
        }
        m++;
    }
    
    unsigned int ignorecode = ~(1 << parttype);
    
    if(filename == ""){
        printUsage(argv[0]);
        exit(1);
    }
    
    
    
    velbins = new double[numbins];
    massbins = new double[numbins];
    countbins = new double[numbins];
    densbins = new double[numbins];
    
    for(int i = 0; i < numbins; i++){
        velbins[i] = 0;
        countbins[i] = 0;
        massbins[i] = 0;
        densbins[i] = 0;
    }
    
    
    GSnap snap(filename, false);
    
    
    if(!snap.IsBlock("POS ")){
        printf("No Position Block!\n");
        exit(1);
    }
    if(!snap.IsBlock("VEL ")){
        printf("No Velocity Block!\n");
        exit(1);
    }
    
    int64_t numTotalParts = snap.GetNpart(parttype);
    //int gridsize = pow(numTotalParts, 1/3);
    double mass = snap.GetHeader().mass[1];
    
    fprintf(stderr, "Filename: %s\n", filename.c_str());
    fprintf(stderr, "Bins: %d\n", numbins);
    fprintf(stderr, "Radius: %f \n", radius);
    fprintf(stderr, "x y z: %f %f %f \n", x, y, z);
    fprintf(stderr, "Num of Particles: %lld\n", numTotalParts);
    fprintf(stderr, "BoxSize: %f\n", snap.GetHeader().BoxSize);
    fprintf(stderr, "Num Files: %d\n", snap.GetNumFiles()); 
    
    dr = radius / numbins;
    
    double totalmass = 0.0; 

    int64_t cts = 0;
    while(cts < numTotalParts){
        vector<float> temppos = snap.GetBlock("POS ", bufferSize, cts, ignorecode);
        vector<float> tempvel = snap.GetBlock("VEL ", bufferSize, cts, ignorecode);
        cts += temppos.size();
        
        //printf("%d\n", cts);
        for(int i = 0; i < temppos.size(); i++){
            //printf("Ok %d\n", i);
            double rx = temppos[3 * i + 0] - x;
            double ry = temppos[3 * i + 1] - y;
            double rz = temppos[3 * i + 2] - z;
           
            double r = sqrt(rx * rx + ry * ry + rz * rz);
            rx /= r;
            ry /= r;
            rz /= r;
            
            //printf("%f %f %f %f\n", rx, ry, rz, r);
            if((r < radius)){
                int ind = r / dr;
                //printf("ind %d\n", ind);
                velbins[ind] += mass *
                        (tempvel[3 * i + 0] * rx +
                         tempvel[3 * i + 1] * ry +
                         tempvel[3 * i + 2] * rz);
                
                countbins[ind] ++;
                massbins[ind] += mass;
                densbins[ind] += mass / (4 * PI / 3
                                              * (pow((ind+1.0) * dr, 3)
                                                 - pow((ind) * dr, 3)));
                
                totalmass += mass;
            }
        }
    }
    
    
    printf("Radius, Velocity, Density, Mass, Counts\n");
    for(int i = 0; i < numbins; i++){
        if(massbins[i] != 0){
            velbins[i] /= massbins[i];
            //densbins[i] = massbins[i] / (4 * PI / 3) / (pow(i * dr + dr, 3) - pow(i * dr, 3));
        }else{
            velbins[i] = 0.0;
            //densbins[i] = 0.0;
        }
        
        printf("%f %e %e %e %d\n", i * dr, velbins[i], densbins[i], massbins[i], (int)countbins[i]);
    }
    
    delete[] velbins;
    delete[] countbins;
    delete[] massbins;
    delete[] densbins;
    
    return 0;
}
