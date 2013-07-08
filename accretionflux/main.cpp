#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "accretion.h"


using namespace std;


//string filename_ =  "/Users/lyang/data/run_200";
//string filename_ =  "/Users/lyang/data/32Mpc_S1_PM_000";
Point halocenter;
double r = 10000;
double dr = 100;

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

int main(){
    
    //string filename_ =  "/Users/lyang/data/trygad.data";
    string filename_ = "/Users/lyang/Documents/Projects/sf13/denscompare/try.data";
    
    GSnap * gsnap_ = new GSnap(filename_, isHighMem, parttype, datagridsize);
    int numparts = gsnap_->Npart;
    double mass = gsnap_->header.mass[1];

    Point * pos = gsnap_ -> getAllPos();//new Point[numparts];
    Point * vel = gsnap_ -> getAllVel();//new Point[numparts];
   
    /*for(int i = 0; i < numparts; i++){
        printf("Pos: %f %f %f\n", pos[i].x, pos[i].y, pos[i].z);
        printf("Vel: %f %f %f\n", vel[i].x, vel[i].y, vel[i].z);
    }*/



    halocenter.x = 16000;
    halocenter.y = 16000.0;
    halocenter.z = 16000.0;
    

    double ar_sphere;
    
    //printf("Measure in radius %f, with thickness %f. Accretion rate is %e\n",
    //       r, dr, ar_sphere);
    //std::cout.flush();
    
    TetraStreamer streamer(filename_,
                           inputmemgrid,
                           parttype,
                           datagridsize,
                           isHighMem,
                           isAllData,
                           isVelocity,
                           true,
                           isInOrder);
    
    double ar_tetra;/* = accretion_tetra_rate(
                    streamer,  
                    mass, 
                    halocenter,
                    r);*/


    //printf("Measure in LTFE %f. Area is %e. Accretion rate is %e\n",
    //               r, 4*PI*r*r, ar_tetra);
    
    
    //test
    //double rho = numparts * mass /
    //    ((gsnap_->header.BoxSize)*(gsnap_->header.BoxSize)*(gsnap_->header.BoxSize));
    //printf("Test:\n, Calculated rate is rho * v * 4 pi r^2 = %e\n", - rho * r * 4 * 3.1415926 * r * r); //* (1/2.0 + 1/2.0 / 10.0));

    
    dr = 10;
    printf("Artifitial data: \n");
    
    printf("Particle Numbers: %d\n", numparts);
    printf("Particle mass: %f\n", mass);

    
    printf("r, Accurate, LTFE, Sphere ");
    for(int i = 1; i <= 20; i +=2){
        printf("dr = %3.0f ", dr * i);
    }
    printf("\n");
    /*for(r = 100.0; r < 20000; r+=1000){
        double accurate_accr =  - rho * r * 4 * 3.1415926 * r * r;
        
        streamer.reset();
        ar_tetra = accretion_tetra_rate(
                  streamer,
                  mass,
                  halocenter,
                  r);
        
        
        printf("%f %e %e ", r, accurate_accr, ar_tetra);
        for(int i = 1; i <= 20; i +=2){
            ar_sphere = accretion_sphere_rate(
                                              numparts,
                                              pos,
                                              vel,
                                              mass,
                                              halocenter,
                                              r,
                                              r + dr * i);
            printf(" %e ", ar_sphere);
            cout.flush();
        }
        printf("\n");

    }*/
    
    
    delete gsnap_;
    
    
    printf("Real data: \n");
    filename_ = "/Users/lyang/data/run_200";
    
    
    halocenter.x = 28464.0;
    halocenter.y = 15309.0;
    halocenter.z = 25433.7;
    
    
    GSnap * gsnap1_ = new GSnap(filename_, isHighMem, parttype, datagridsize);
    numparts = gsnap1_->Npart;
    mass = gsnap1_->header.mass[1];
    printf("Particle Numbers: %d\n", numparts);
    printf("Particle mass: %f\n", mass);
    
    pos = gsnap1_ -> getAllPos();//new Point[numparts];
    vel = gsnap1_ -> getAllVel();//new Point[numparts];
    
    TetraStreamer streamer1(filename_,
                           inputmemgrid,
                           parttype,
                           datagridsize,
                           isHighMem,
                           isAllData,
                           isVelocity,
                           true,
                           isInOrder);
    
    
    
    printf("r, Accurate, LTFE, Sphere ");
    for(int i = 1; i <= 20; i +=2){
        printf("dr = %3.0f ", dr * i);
    }
    printf("\n");
    for(r = 5500; r < 6500; r+=100){
        
        streamer1.reset();
        ar_tetra = accretion_tetra_rate(
                                        streamer1,
                                        mass,
                                        halocenter,
                                        r);
        
        printf("%f %e ", r, ar_tetra);
        for(int i = 1; i <= 20; i +=2){
            ar_sphere = accretion_sphere_rate(
                                              numparts,
                                              pos,
                                              vel,
                                              mass,
                                              halocenter,
                                              r,
                                              r + dr * i);
            printf(" %e ", ar_sphere);
            cout.flush();
        }
        printf("\n");
        
    }
    
    delete gsnap1_;
    
    return 0;
}
