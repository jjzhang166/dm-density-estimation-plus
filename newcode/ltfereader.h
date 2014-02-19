/*
 This class is for reading the LTFE density rendering.
 Author: Lin F. Yang
 Copyright reserved
 */

#ifndef __LY_LTFEREADER__
#define __LY_LTFEREADER__

class LTFEReader{
public:
    LTFEReader(std::string filename);
    ~LTFEReader();
    LTFEHeader getHeader();
    float * getData();
    std::vector<float> & getDataVec();
private:
    std::string filename_;
    
    //float * data_;
    std::vector<float> * dataVec_;
    LTFEHeader header_;
    
};

#endif