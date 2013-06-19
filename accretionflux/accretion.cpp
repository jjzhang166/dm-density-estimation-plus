#include <cmath>
#include "accretion.h"
#include "triangle.h"
#include "tetracut.h"

using namespace std;

double accretion_sphere_rate(int counts, Point * posdata, Point * veldata, double mass, Point &halocenter, double r1, double r2){
    double volum = 4.0 * PI / 3.0 * (r2 * r2 * r2 - r1 * r1 * r1);
    double accr = 0.0;
    double dr = abs(r2 - r1);
    //printf("Points %d %f\n", counts, 1.0/volum);
    for(int i = 0; i < counts; i ++){
        //flag the particles
        if(posdata[i].x < 0) continue;
        
        Point rvec = halocenter - posdata[i];
        double r = sqrt(rvec.dot(rvec));
        rvec = rvec / r;
        if(r > r1 && r < r2){
            //printf("%f\n", r);
            accr += rvec.dot(veldata[i]);
        }
    }
    return accr * mass / dr;// * 4.0 * PI * r1 * r1 / volum;
}

double getRadius(const Point &v1, const Point &v2){
    Point rvec = v1-v2;
    return sqrt(rvec.dot(rvec));
}

double accretion_tetra_rate(
                    TetraStreamer &tetrastreamer, 
                    double mass, 
                    Point &halocenter,
                    double r){
    int cts = 0;
    double accr = 0.0;
    tetrastreamer.reset();
    IsoCutter cutter;
    while(tetrastreamer.hasNext()){
        Tetrahedron * tetras;
        int nums;
        tetras = tetrastreamer.getNext(nums);
        
        for(int i = 0; i < nums; i++){
            
            //printf("%f\n", tetras[i].invVolume);
            
            cutter.setTetrahedron(&tetras[i]);
            cutter.setValues(getRadius(tetras[i].v1, halocenter),
                             getRadius(tetras[i].v2, halocenter),
                             getRadius(tetras[i].v3, halocenter),
                             getRadius(tetras[i].v4, halocenter)
                            );
            
            int nu_tri = cutter.cut(r, tetras[i].velocity1, tetras[i].velocity2, tetras[i].velocity3, tetras[i].velocity4);
            
            /*if(nu_tri > 0){
                printf("Radius: %d %f %f %f %f %f\n", nu_tri, r,
                            getRadius(tetras[i].v1, halocenter),
                            getRadius(tetras[i].v2, halocenter),
                            getRadius(tetras[i].v3, halocenter),
                            getRadius(tetras[i].v4, halocenter));
            }*/

            cts ++;
            for(int j = 0; j < nu_tri; j++){
                Triangle3d t = cutter.getTrangle(j);
                Point direc = halocenter - (t.a + t.b + t.c) / 3.0 ;
                
                double rp = sqrt(direc.dot(direc));
                
                //printf("%f\n", sqrt(direc.dot(direc)));
                direc = direc / rp;
                
                double velocity = direc.dot(t.val1 + t.val2 + t.val3) / 3.0;
                //printf("%e\n", velocity);
                double area = t.getArea();// / (r*r);
                //printf("\n");
                /*printf("%f %f %f\n", t.a.x, t.a.y, t.a.z);
                printf("%f %f %f\n", t.b.x, t.b.y, t.b.z);
                printf("%f %f %f\n", t.c.x, t.c.y, t.c.z);
                
               
                printf("r=%f, dis=%f, ratio=%f\n", r, rp, rp/r);
                
                printf("%f %f %f\n", t.val1.x, t.val1.y, t.val1.z);
                printf("%f %f %f\n", t.val2.x, t.val2.y, t.val2.z);
                printf("%f %f %f\n", t.val3.x, t.val3.y, t.val3.z);*/

                accr += area * velocity * mass * tetras[i].invVolume / 6.0;
                //printf("Area: %f %f\n", area, area/(4*PI*r*r));
                //printf("\n");
                

            }
        }
    }
    printf("Tetras: %d\n", cts);
    return accr;
}

