#ifndef __LYTYPE__
#define __LYTYPE__

#define PI 3.14159265358979323846

//use double as position variables
//#define __POS_DOUBLE

//use 64 bit integers as idtype
//#define __ID_UINT64

#ifdef __POS_DOUBLE
#define REAL double
#else
#define REAL float
#endif

#ifdef __ID_UINT64
#define IDTYPE uint64_t
#else
#define IDTYPE uint32_t
#endif


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
