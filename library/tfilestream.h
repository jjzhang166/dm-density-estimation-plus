#ifndef __TFILE_STREAM_LY
#define __TFILE_STREAM_LY
#include <fstream>
#include <string>
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
    int getNumofTetras();

    void reset();

private:
    int tetrablock_;
    fstream inputfile_;
    std::string filename_;
    TFileHeader header_;
    Tetrahedron * blockTetras_;
    int currentBlockID_;
    int currentNumOfTetras_;
    int blocksize_;
};



#endif