#include <cmath>
#include <algorithm>    // std::min
#include "viewfun.h"

using namespace std;

#define MINVAL 1e-3

void getWarmColor(float v, float &r, float &g, float &b){
    r = -1.8966 * v*v*v + 1.2049 * v*v + 1.1463 * v + 0.2253;
    g = -2.713 * v*v + 2.5221 * v + 0.2499;
    b = 2.3924 * v*v*v - 5.264 * v*v + 2.2936 * v + 0.7214;
    
    if(r > 1.0) r = 1.0;
    if(r < 0.0) r = 0.0;
    if(g > 1.0) g = 1.0;
    if(g < 0.0) g = 0.0;
    if(b > 1.0) b = 1.0;
    if(b < 0.0) b = 0.0;
}


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

void getcolorImge(float *value, float * colorimg, int numpixels){
    float r, g, b;
    float max_ = -1.0e30, min_ = 1.0e30;
    for(int i = 0; i < numpixels; i++){
        if(max_ < abs(value[i])){
            max_ = abs(value[i]);
        }
        if(min_ > abs(value[i])){
            min_ = abs(value[i]);
        }
        
        //printf("%e\n", value[i]);
        
    }
    
    //shift the min to 1e-5
    float shifts = 0;
    
    //if(min_ < 0){
    //    shifts = MINVAL - min_;
    //}
    
    min_ =  min_ + shifts;
    max_ = max_ + shifts;
    
	//printf("%f %f\n", value[i]);
	if(min_ == max_){
		min_ = max_ * MINVAL;
	}
    
    
    float x = log(max_) - log(min_);
    for(int i = 0; i < numpixels; i++){
        float v = (log(abs(value[i]) + shifts) - log(min_)) / x;
        //printf("%f\n",v);
        getJetColor(v, r, g, b);
        colorimg[3 * i] = r;
        colorimg[3 * i + 1] = g;
        colorimg[3 * i + 2] = b;
    }
}
