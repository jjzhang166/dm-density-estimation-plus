#include <string>
#include <cstdlib>
#include <cmath>
#include <stdint.h>
#include <cstdio>
#include <vector>
#include <fstream>
#include <sstream>
#include "io_utils.h"

using namespace std;


void writeToDividerFile(string outputBaseName_,
                        int type,
                        int i,
                        ios_base::openmode mode,
                        const char* s,
                        streamsize n,
                        bool isHeader
                        ){
    stringstream ss;
    ss << i;
    string filename = "";
    
    
    //printf("File id: %d\n", i);
    //printf("%d\n", i);
    
    if(type == INDEXTYPE){
        filename = outputBaseName_ + "."INDEXSUFFIX"." + ss.str();
    }else if(type == POSTYPE){
        filename = outputBaseName_ + "."POSSUFFIX"." + ss.str();
    }else if(type == VELTYPE){
        filename = outputBaseName_ + "."VELSUFFIX"." + ss.str();
    }else{
        return;
    }
    
    fstream outDataStream;
    outDataStream.open(filename.c_str(), mode);
    
    if(isHeader){
        outDataStream.seekp(0, ios::beg);
    }
    
    if(!outDataStream.good()){
        printf("Bad file: %s!\n", filename.c_str());
        exit(1);
    }
    outDataStream.write(s, n);
    outDataStream.close();
    
}


divide_header readDividerFileHeader(std::string basename, int i){
    fstream infileStream;
    stringstream ss;
    ss << i;
    string firstfile = basename + "."INDEXSUFFIX"." + ss.str();
    infileStream.open(firstfile.c_str(), ios::in | ios::binary);
    divide_header header;
    if(!infileStream.good()){
        fprintf(stderr, "File Error: %s!\n", basename.c_str());
        exit(1);
    }
    infileStream.read((char *) &header, sizeof(divide_header));
    infileStream.close();
    return header;
}

int getNumDividerFiles(std::string basename){
    divide_header header = readDividerFileHeader(basename, 0);
    return header.totalfiles;
}

int64_t getDividerFileNumParts(std::string basename, int i){
    divide_header header = readDividerFileHeader(basename, i);
    return header.numparts;
}

int64_t readDividerFileInds(std::string basename, int i, void * inds){
    fstream infileStream;
    stringstream ss;
    ss << i;
    string filename = basename + "."INDEXSUFFIX"." + ss.str();
    
    infileStream.open(filename.c_str(), ios::in | ios::binary);
    divide_header header;
    if(!infileStream.good()){
        fprintf(stderr, "File Error: %s!\n", filename.c_str());
        exit(1);
    }
    infileStream.read((char *) &header, sizeof(divide_header));
    infileStream.read((char *) inds, sizeof(int64_t) * header.numparts);
    infileStream.close();
    
    return header.numparts;
}

int64_t readDividerFilePos(std::string basename, int i, void * pos){
    fstream infileStream;
    stringstream ss;
    ss << i;
    string filename = basename + "."POSSUFFIX"." + ss.str();
    
    infileStream.open(filename.c_str(), ios::in | ios::binary);
    divide_header header;
    if(!infileStream.good()){
        fprintf(stderr, "File Error: %s!\n", filename.c_str());
        exit(1);
    }
    infileStream.read((char *) &header, sizeof(divide_header));
    infileStream.read((char *) pos, sizeof(float) * 3 * header.numparts);
    infileStream.close();
    return header.numparts;
}


int64_t readDividerFileVel(std::string basename, int i, void * vel){
    fstream infileStream;
    stringstream ss;
    ss << i;
    string filename = basename + "."VELSUFFIX"." + ss.str();
    
    infileStream.open(filename.c_str(), ios::in | ios::binary);
    divide_header header;
    if(!infileStream.good()){
        fprintf(stderr, "File Error: %s!\n", filename.c_str());
        exit(1);
    }
    infileStream.read((char *) &header, sizeof(divide_header));
    infileStream.read((char *) vel, sizeof(float) * 3 * header.numparts);
    infileStream.close();
    return header.numparts;
}
