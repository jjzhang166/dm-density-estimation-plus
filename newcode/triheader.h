#ifndef __TRIHEADER__
#define __TRIHEADER__
#include <stdint.h>

#define VELXFILESUFFIX "velx"
#define VELYFILESUFFIX "vely"
#define VELZFILESUFFIX "velz"
#define DENFILESUFFIX "den"
#define TRIFILESUFFIX "tri"

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

class TriHeader{
public:
    int32_t NumBlocks;      //how many blocks in this file
    int64_t TotalTriangles; //total trianles in this file
    int32_t ImageSize;      //what's the image size of this cut
	double boxSize;         //what's the box size
    double startZ;          //what's the star Z coordinates
    double dz;              //what's the end Z coordinates
    int numOfZPlanes;    //how many z planes are there
    char other[256 - 4 - 8 - 4 - 8 * 3 - 4];
};

class TriBlockHeader{
public:
    int64_t TotalTriangles; //total trianles in this file
    char other[256 - 8];
};

#endif