#ifndef _LY_DEFINES__
#define _LY_DEFINES__

#define INDEXSUFFIX "ind"
#define INDEXTYPE 0
#define POSSUFFIX "pos"
#define POSTYPE 1
#define VELSUFFIX "vel"
#define VELTYPE 2

//#define LONGINDS

typedef float  float32_t;
typedef double float64_t;

struct divide_header{
    int32_t filenumber;
    int32_t totalfiles;
    int32_t gridsize;
    int32_t numofZgrids;
    int64_t startind;
    int64_t numparts;
    float64_t boxSize;
    char unused[256 - 4 * 4 - 8 * 3];
};

//write the data to devider file
void writeToDividerFile(std::string outputBaseName_,
                 int type,
                 int i,
                 std::ios_base::openmode mode,
                 const char* s,
                 std::streamsize n,
                 bool isHeader = false
                 );

int getNumDividerFiles(std::string basename);
divide_header readDividerFileHeader(std::string basename, int i);
int64_t getDividerFileNumParts(std::string basename, int i);
//read data into inds
int64_t readDividerFileInds(std::string basename, int i, void * inds);
int64_t readDividerFilePos(std::string basename, int i, void * pos);
int64_t readDividerFileVel(std::string basename, int i, void * vel);
#endif