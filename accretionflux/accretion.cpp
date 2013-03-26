#include <cmath>
#include "accretion.h"

using namespace std;

double accreation_sphere_rate(int counts, Point * posdata, Point * veldata, double mass, Point &halocenter, double r1, double r2){
    double volum = 4.0 * PI / 3.0 * (r2 * r2 * r2 - r1 * r1 * r1);
    double accr = 0.0;
    for(int i = 0; i < counts; i ++){
        Point rvec = halocenter - posdata[i];
        double r = sqrt(rvec.dot(rvec));
        rvec = rvec / r;
        if(r > r1 && r < r2){
            accr += rvec.dot(veldata[i]);
        }
    }
    return accr * mass / volum * 4.0 * PI * r1 * r1;
}

