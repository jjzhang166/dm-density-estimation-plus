#ifndef __TFILE_STREAM_LY
#define __TFILE_STREAM_LY
#include <fstream>
#include <string>
#include <stdint.h>
#include "tfileheader.h"
#include "tetrahedron.h"
class TFileStream{
public:
    //load how many tetrahdrons each time
    TFileStream(std::string filename, int blocksize);
    ~TFileStream();

    bool hasNext();
    Tetrahedron * getNext(int & numtetras);
    
    TFileHeader getHeader();
    uint64_t getNumofTetras();

    void reset();

private:
    //long tetrablock_;
    fstream inputfile_;
    std::string filename_;
    TFileHeader header_;
    Tetrahedron * blockTetras_;
    uint64_t currentBlockID_;
    int currentNumOfTetras_;
    int blocksize_;
};



#endif

