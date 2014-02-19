#ifndef TRIFILEUTIL
#define TRIFILEUTIL
#include <string>
#include <stdint.h>
#include <fstream>
#include "triheader.h"

#ifdef __GNUC__
#define DEPRECATED(func) func __attribute__ ((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED(func) __declspec(deprecated) func
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED(func) func
#endif


#define TRIFILESUFFIX "tri"
#define VERTEXFILESUFFIX "vert"
#define DENSITYFILESUFFIX "dens"
#define VELXFILESUFFIX "velx"
#define VELYFILESUFFIX "velx"
#define VELZFILESUFFIX "vely"

class TrifileWriter{
public:
    TrifileWriter(TriHeader header, bool isVelocity=false);
    ~TrifileWriter();
    void open(std::string basename);
    bool isOpen();
    bool good();
    

    
    // the following functionis deprecated
    DEPRECATED(void write(int * trianglesPerPlane,
               std::vector<int> & trianglePlaneIds_,
               std::vector<float> & vertexData_,
               std::vector<float> & densityData_));
    
    
    void write(int * trianglesPerPlane,
               std::vector<int> * trianglePlaneIds_,
               std::vector<float> * vertexData_,
               std::vector<float> * densityData_);
    
    void write(int * trianglesPerPlane,
               std::vector<int> * trianglePlaneIds_,
               std::vector<float> * vertexData_,
               std::vector<float> * densityData_,
               std::vector<float> * velXData_,
               std::vector<float> * velYData_,
               std::vector<float> * velZData_);
    
    void close();
    
private:
    TriHeader header_;
    std::string basename_;
    std::fstream vertexFileStream_;
    std::fstream densityFileStream_;
    
    std::fstream velxFileStream_;
    std::fstream velyFileStream_;
    std::fstream velzFileStream_;
    
    int * numTrianglesPerPlane_;
    int * numTrianglesPerPlaneCurrentBlock_;
    float * zCoorPlane_;
    int numBlocks_;
    
    // for sorting the vectors
    int * outputinds;
    
    // use only once before writting
    // trianglesPerPlane: how many triangles for each plane
    // trianglePlaneIds: what is the PlaneID for each triangle
    void setTrisPerPlane(int * trianglesPerPlane,
                         std::vector<int> &trianglePlaneIds_);
    bool isVelocity_;
    
    float * densSorted;
    float * vertexSorted;
    float * velxSorted;
    float * velySorted;
    float * velzSorted;
    
    int cBufferSize_;
};


// TODO
class TrifileReader{
public:
    TrifileReader(std::string basename, bool isVelocity = false);
    ~TrifileReader();
    
    TriHeader getHeader(){
        return header_;
    };
    
    bool isOpen();
    void close();
    
    float getZcoor(int plane);
    float getNumTriangles(int plane);
    
    void loadPlane(int plane);
    float * getTriangles();
    float * getDensity();
    float * getVelocityX();
    float * getVelocityY();
    float * getVelocityZ();
    
    
private:

    
    TriHeader header_;
    std::string basename_;
    
    std::fstream vertexFileStream_;
    std::fstream densityFileStream_;
    std::fstream velxFileStream_;
    std::fstream velyFileStream_;
    std::fstream velzFileStream_;
    
    int * numTrianglesPerPlane_;
    int * numTrianglesPerPlaneCurrent_;
    float * zCoorPlane_;
    int numBlocks_;
    bool isVelocity_;
    
    //std::vector<float> vertexData_;
    //std::vector<float> densityData_;
    float * vertexData_;
    float * densityData_;
    float * velXData_;
    float * velYData_;
    float * velZData_;
    
};
#endif