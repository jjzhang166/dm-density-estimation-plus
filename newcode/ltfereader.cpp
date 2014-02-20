#include <fstream>
#include <stdint.h>
#include <vector>
#include "ltfeheader.h"
#include "ltfereader.h"


using namespace std;

LTFEReader::LTFEReader(std::string filename){
    filename_ = filename;
    
    fstream LTFEStream(filename.c_str(), ios::in | ios::binary);
    
    if(!LTFEStream.good()){
        fprintf(stderr, "File Error!\n");
        exit(1);
    }
    
    LTFEStream.read((char *) &header_, sizeof(LTFEHeader));
    
    int64_t numparts = header_.xyGridSize * header_.xyGridSize * header_.zGridSize;
    //data_ = new float[numparts];
    dataVec_ = new vector<float>(numparts);
    
    LTFEStream.read((char *) &(dataVec_->front()), sizeof(float) * numparts);
    
    LTFEStream.close();
}

LTFEHeader LTFEReader::getHeader(){
    return header_;
}

float * LTFEReader::getData(){
    return &(dataVec_->front());
}

std::vector<float> & LTFEReader::getDataVec(){
    return *dataVec_;
}

LTFEReader::~LTFEReader(){
    //delete data_;
    delete dataVec_;
}