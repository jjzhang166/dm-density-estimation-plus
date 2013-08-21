#include <fstream>
#include "tfilestream.h"

using namespace std;

TFileStream::TFileStream(std::string filename, int blocksize){
    filename_ = filename;
    inputfile_.open(filename.c_str(), ios::in | ios::binary);
    blocksize_ = blocksize;
    blockTetras_ = new Tetrahedron[blocksize];
    currentBlockID_ = 0;
    if(inputfile_.good()){
        inputfile_.read((char *)&header_, sizeof(TFileHeader));
    }else{
        printf("File Corrupted!\n");
        exit(1);
    }
}

TFileStream::~TFileStream(){
    delete blockTetras_;
    inputfile_.close();
}

TFileHeader TFileStream::getHeader(){
    return header_;
}

int TFileStream::getNumofTetras(){
    return header_.numOfTetrahedrons;
}

bool TFileStream::hasNext(){
    if(currentBlockID_ * blocksize_ < (int)header_.numOfTetrahedrons){
        return true;
    }else{
        return false;
    }
}

Tetrahedron * TFileStream::getNext(int & numtetras){
    if(hasNext()){
        int numts = getNumofTetras() - currentBlockID_ * blocksize_;
        if(numts >= blocksize_){
            numts = blocksize_;
        }
        
        currentBlockID_ ++;
        currentNumOfTetras_ = numts;
        numtetras = currentNumOfTetras_;
        
        inputfile_.read((char *) blockTetras_,
                        sizeof(Tetrahedron) * currentNumOfTetras_);
    }else{
        currentNumOfTetras_ = 0;
        numtetras = 0;
    }
    return blockTetras_;
}


void TFileStream::reset(){
    inputfile_.seekg (sizeof(TFileHeader), inputfile_.beg);
    currentBlockID_ = 0;
    currentNumOfTetras_ = 0;
}