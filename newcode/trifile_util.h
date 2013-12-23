#ifndef TRIFILEUTIL
#define TRIFILEUTIL
#include <string>
#include <stdint.h>
#include <fstream>
#include "triheader.h"

#define TRIFILESUFFIX "tri"
#define VERTEXFILESUFFIX "vert"
#define DENSITYFILESUFFIX "dens"

class TrifileWriter{
public:
    TrifileWriter(TriHeader header);
    ~TrifileWriter();
    void open(std::string basename);
    bool isOpen();
    bool good();
    void write(int * trianglesPerPlane,
               std::vector<int> & trianglePlaneIds_,
               std::vector<float> & vertexData_,
               std::vector<float> & densityData_);
    void close();
    
private:
    TriHeader header_;
    std::string basename_;
    std::fstream vertexFileStream_;
    std::fstream densityFileStream_;
    
    int * numTrianglesPerPlane_;
    int * numTrianglesPerPlaneCurrentBlock_;
    float * zCoorPlane_;
    int numBlocks_;
};


class TrifileReader{
public:
    TrifileReader(std::string basename);
    ~TrifileReader();
    
    TriHeader getHeader(){
        return header_;
    };
    
    bool isOpen();
    void close();
    
    float getZcoor(int plane);
    float getNumTriangles(int plane);
    
    void loadPlane(int plane);
    float * getTriangles(int plane);
    float * getDensity(int plane);
    
private:

    
    TriHeader header_;
    std::string basename_;
    std::fstream vertexFileStream_;
    std::fstream densityFileStream_;
    
    int * numTrianglesPerPlane_;
    int * numTrianglesPerPlaneCurrent_;
    float * zCoorPlane_;
    int numBlocks_;
    
    //std::vector<float> vertexData_;
    //std::vector<float> densityData_;
    float * vertexData_;
    float * densityData_;
};
#endif