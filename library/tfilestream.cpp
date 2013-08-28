#include <cstdlib>
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
        fprintf(stderr,"File Corrupted!\n");
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

uint64_t TFileStream::getNumofTetras(){
    return header_.numOfTetrahedrons;
}

bool TFileStream::hasNext(){
    if((uint64_t)currentBlockID_ * (uint64_t)blocksize_ < (uint64_t)header_.numOfTetrahedrons){
        return true;
    }else{
        return false;
    }
}

Tetrahedron * TFileStream::getNext(int & numtetras){
    if(hasNext()){
        uint64_t numts = getNumofTetras() - currentBlockID_ * blocksize_;
        if(numts >= (uint64_t) blocksize_){
            numts = (uint64_t) blocksize_;
        }
        
        currentBlockID_ ++;
        currentNumOfTetras_ = numts;
        numtetras = (int) currentNumOfTetras_;
        
        inputfile_.read((char *) blockTetras_,
                        sizeof(Tetrahedron) * currentNumOfTetras_);
    }else{
        currentNumOfTetras_ = 0;
        numtetras = 0;
    }
    return blockTetras_;
}


void TFileStream::reset(){
	if(inputfile_.eof()){
		inputfile_.clear();
	}
    inputfile_.seekg (sizeof(TFileHeader), inputfile_.beg);
    currentBlockID_ = 0;
    currentNumOfTetras_ = 0;
}

