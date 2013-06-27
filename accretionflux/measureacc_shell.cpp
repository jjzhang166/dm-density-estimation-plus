#include <cstring>
#include <cstdio>
#include <cstdlib>     /* atoi */
#include <fitsio.h>
#include <sstream>
#include <vector>
#include <iostream>
#include "accretion.h"
#include "haloread.h"
#include "../library/types.h"

using namespace std;

void printusage(){
    printf("Usage: measureacc_shell datafile halocatalog\n");
    printf("[-radius <the radius measured in r_vir>]\n");
    printf("[-halo <{ list of halos to measure }>]\n");
    printf("\n");
    printf("Measure the accretion rate of halos in the datafile (gadget file)\n descripted by the halocatalog (fits file)\n");
    printf("The default radius is 1.5\n");
}

//string filename_ =  "/Users/lyang/data/32Mpc_S1_PM_000";
Point halocenter;
double r = 10000;
double dr = r / 100.0;

//load all the particles into memory?
bool isHighMem = true;
//return the all particle pointer?
bool isAllData = false;

//if use -1, then use the particle gridsize as the gridsize
//otherwise use the user setting
int datagridsize = 256;
//the particle type in the gadget file
int parttype = 1;
int inputmemgrid = 32;

bool isInOrder = false;
bool isVelocity = true;


int main(int argc, char *argv[])
{
    string datafile;
    string halofile;
    double radtimes = 1.5;
    std::vector<int> halovec;
    bool isUseHaloVec = false;
    
    if (argc < 3) {
        printusage();
        return 1;
    }else{
        datafile = argv[1];
        halofile = argv[2];
        int k = 3;
        while(k < argc){
            stringstream ss;
            //printf("%s\n", argv[k]);
			if(strcmp(argv[k], "-radius") == 0){
				ss << argv[k + 1];
				k +=2;
                ss >> radtimes;
                //printf("%f\n", radtimes);
			}else if(strcmp(argv[k], "-halo") == 0){
                k ++;
                isUseHaloVec = true;
                if(strcmp(argv[k], "{") != 0){
                    printusage();
                    return 1;
                }else{
                    k ++;
                    while((k < argc) && (strcmp(argv[k], "}") != 0)){
                        int halon;
                        stringstream ss1;
                        ss1 << argv[k];
                        ss1 >> halon;
                        halovec.push_back(halon);
                        k++;
                        //printf("%d\n", halon);
                    }
                    if((k == argc) || (strcmp(argv[k], "}") != 0)){
                        printusage();
                        return 1;
                    }else{
                        k++;
                    }
                }
                
            }else{
                printusage();
                return 1;
            }
        }
    }
    
    
    int hmin =0;
    int hmax = getTotalHaloNum(halofile.c_str());
    if(isUseHaloVec){
        hmin = 0;
        hmax = halovec.size();
    }
    Halo halo;
    
    GSnap * gsnap_ = new GSnap(filename_, isHighMem, parttype, datagridsize);
    numparts = gsnap_->Npart;
    mass = gsnap_->header.mass[1];
    printf("Particle Numbers: %d\n", numparts);
    printf("Particle mass: %f\n", mass);
    
    Point * pos = gsnap_ -> getAllPos();//new Point[numparts];
    Point * vel = gsnap_ -> getAllVel();//new Point[numparts];
    
    printf("ID    MASS    X    Y    Z    RADIUS    ACCRETION_RATE\n");
    
    for(int i = hmin; i < hmax; i++){
        int haloid = 0;
        if(isUseHaloVec){
            haloid = halovec[i];
        }else{
            haloid = i + 1;
        }
        int status = getHaloById(halofile.c_str(), haloid, &halo);
        if(status != 0){
            break;
        }
        
        halocenter.x = halo.x;
        halocenter.y = halo.y;
        halocenter.z = halo.z;
        double mass = halo.mass;
        double r = radtimes * halo.radius;
        
        double accrate = accretion_sphere_rate(
                                               numparts,
                                               pos,
                                               vel,
                                               mass,
                                               halocenter,
                                               r,
                                               r + dr);
        
        printf("%d   %f   %f   %f   %f   %f   %f \n", haloid, mass, halocenter.x,
               halocenter.y, halocenter.z, halo.radius, accrate);
        cout.flush();
    }
    
    delete gsnap_;
    
}
