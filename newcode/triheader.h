#ifndef __TRIHEADER__
#define __TRIHEADER__
#include <stdint.h>

#define TRIFILESUFFIX "tri"
#define VERTEXFILESUFFIX "vert"
#define DENSITYFILESUFFIX "dens"
#define DENFILESUFFIX DENSITYFILESUFFIX
#define VELXFILESUFFIX "velx"
#define VELYFILESUFFIX "vely"
#define VELZFILESUFFIX "velz"

/*******FILE STRUCTURE**********
    <TriHeader>                                     [256 bytes]
    <List of Number of Triangles of each z-plane>   [4*numOfZPlanes bytes]
    <List of z-coor of each z-plane>                [4*numOfZPlanes bytes]
    <BlockHeader0>                                  [256 bytes]
    <List of Number of Triangles of each z-plane>   [4*numOfZPlanes bytes]
    <List Of Triangles>                             [BlockHeader0.NumTriangles*6*4 bytes or BlockHeader0.NumTriangles*4 bytes]
    ....
*******************************/

/*class TriHeader{
public:
    int32_t fileID;
    int32_t NumFiles;
    int64_t NumTriangles;
    int32_t z_id;
	float boxSize;
    double z_coor;
    char other[256 - 4 - 4 - 8 - 4 - 8];
};*/

//type of the files
const int VELX =  0x01;
const int VELY =  0x02;
const int VELZ =  0x04;
const int POS  =  0x08;
const int DENS =  0x10;


class TriHeader{
public:
    int32_t NumBlocks;      //how many blocks in this file
    int32_t ImageSize;      //what's the image size of this cut
    int32_t numOfZPlanes;   //how many z planes are there
    
    int64_t TotalTriangles; //total trianles in this file
	double boxSize;         //what's the box size
    double startZ;          //what's the star Z coordinates
    double dz;              //what's the end Z coordinates
    
    int fileType;           //that is the file type
    char other[256 - (4 * 3 + 8 * 5 + 4)];
};

class TriBlockHeader{
public:
    int64_t TotalTriangles; //total trianles in this file
    char other[256 - 8];
};

#endif