#ifndef __LYTYPE__
#define __LYTYPE__

#define PI 3.14159265358979323846

typedef float REAL;

class Halo{
public:
     float x;
     float y;
     float z;
     float vx;
     float vy;
     float vz;
     float lx;
     float ly;
     float lz;//               
     float i1;//              
     float i2;//              
     float i3;//              
     float i1x;//           
     float i1y;//             
     float i1z;//             
     float i2x;//             
     float i2y;//             
     float i2z;//             
     float i3x;//              
     float i3y;//             
     float i3z;//             
     float mass;//             
     float radius;//          
     int   multi;//          
     float v_max;//           
     float m_max;//           
     float r_max;//           
     int parent_id;//       
     float parent_common;
};
#endif
