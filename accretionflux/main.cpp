#include "accretion.h"

using namespace std;

string filename_ =  "/Users/lyang/data/run_200";
Point halocenter;
double r = 600;
double dr = 10;

int main(){ 
    GSnap * gsnap_ = new GSnap(filename_); 
    int numparts = gsnap_->Npart;
    double mass = gsnap_->header.mass[1];
    printf("Particle Numbers: %d\n", numparts);
    printf("Particle mass: %f\n", mass);
    fstream file(filename_.c_str(), ios_base::in | ios_base::binary);
    if(!file.good()){
        printf("Data file bad!\n");
        exit(1);
    }

    Point * pos = new Point[numparts];
    Point * vel = new Point[numparts];

    gsnap_->readPos(file, pos, 0, numparts);
    gsnap_->readVel(file, vel, 0, numparts);
   
    //for(int i = 0; i < numparts; i++){
    //    printf("Pos: %f %f %f\n", pos[i].x, pos[i].y, pos[i].z);
    //    printf("Vel: %f %f %f\n", vel[i].x, vel[i].y, vel[i].z);
    //}



    halocenter.x = 28464.0;
    halocenter.y = 15309.0;
    halocenter.z = 25433.7;
    

    double ar_sphere = accretion_sphere_rate(
                    numparts, 
                    pos, 
                    vel, 
                    mass, 
                    halocenter,
                    r, 
                    r+dr);
    
    TetraStreamer streamer(filename_, 70, true, true, false);
    double ar_tetra = accretion_tetra_rate( 
                    streamer,  
                    mass, 
                    halocenter,
                    r);

    printf("Measure in radius %f, with thickness %f. Accretion rate is %e\n", 
                    r, dr, ar_sphere);
    printf("Measure in LTFE %f. Area is %f. Accretion rate is %e\n", 
                    r, 4*PI*r*r, ar_tetra);


    file.close();
    
    
    return 0;
}
