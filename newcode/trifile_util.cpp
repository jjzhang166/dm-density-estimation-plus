#include <cstring>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include "trifile_util.h"


using namespace std;

TrifileWriter::TrifileWriter(TriHeader header, bool isVelocity){
    header_ = header;
    header_.NumBlocks = 0;
    header_.TotalTriangles = 0;
    numTrianglesPerPlane_ = new int[header.numOfZPlanes];
    numTrianglesPerPlaneCurrentBlock_ = new int[header.numOfZPlanes];
    
    memset(numTrianglesPerPlane_, 0, sizeof(int) * header.numOfZPlanes);
    
    zCoorPlane_ = new float[header.numOfZPlanes];
    for(int i = 0; i < header.numOfZPlanes; i++){
        zCoorPlane_[i] = header.startZ + header.dz * i;
    }
    
    numBlocks_ = 0;
    basename_ = "";
    
    isVelocity_ = isVelocity;
    
    
    cBufferSize_ = 0;
    
    densSorted = NULL;
    vertexSorted = NULL;
    velxSorted = NULL;
    velySorted = NULL;
    velzSorted = NULL;
    outputinds = NULL;
}

TrifileWriter::~TrifileWriter(){
    close();
    
    delete[] numTrianglesPerPlane_;
    delete[] zCoorPlane_;
    delete[] numTrianglesPerPlaneCurrentBlock_;
    
    if(outputinds != NULL){
        free(outputinds);
    }
    if(densSorted != NULL){
        free(densSorted);
    }
    if(vertexSorted != NULL){
        free(vertexSorted);
    }
    if(velxSorted != NULL){
        free(velxSorted);
    }
    if(velySorted != NULL){
        free(velySorted);
    }
    if(velzSorted != NULL){
        free(velzSorted);
    }
}


//open the file and prepare the file for writing
void TrifileWriter::open(std::string basename){
    
    string vertfn = basename + "." TRIFILESUFFIX "." VERTEXFILESUFFIX;
    string densfn = basename + "." TRIFILESUFFIX "." DENSITYFILESUFFIX;
    string velxfn = basename + "." TRIFILESUFFIX "." VELXFILESUFFIX;
    string velyfn = basename + "." TRIFILESUFFIX "." VELYFILESUFFIX;
    string velzfn = basename + "." TRIFILESUFFIX "." VELZFILESUFFIX;
    
    vertexFileStream_.open(vertfn.c_str(), ios::binary | ios::out);
    densityFileStream_.open(densfn.c_str(), ios::binary | ios::out);
    

    
    vertexFileStream_.write((char *) & header_, sizeof(header_));
    densityFileStream_.write((char *) & header_, sizeof(header_));
    
    /*printf("init ok, %d %d %d %d %d \n", header_.numOfZPlanes, numTrianglesPerPlane_[0], numTrianglesPerPlane_[1], numTrianglesPerPlane_[2], numTrianglesPerPlane_[3]);*/
    
    vertexFileStream_.write((char *) numTrianglesPerPlane_, sizeof(int) * header_.numOfZPlanes);
    vertexFileStream_.write((char *) zCoorPlane_, sizeof(float) * header_.numOfZPlanes);
    
    
    densityFileStream_.write((char *) numTrianglesPerPlane_, sizeof(int) * header_.numOfZPlanes);
    densityFileStream_.write((char *) zCoorPlane_, sizeof(float) * header_.numOfZPlanes);
    
    
    if(isVelocity_){
        
        velxFileStream_.open(velxfn.c_str(), ios::binary | ios::out);
        velyFileStream_.open(velyfn.c_str(), ios::binary | ios::out);
        velzFileStream_.open(velzfn.c_str(), ios::binary | ios::out);
        
        velxFileStream_.write((char *) & header_, sizeof(header_));
        velxFileStream_.write((char *) numTrianglesPerPlane_, sizeof(int) * header_.numOfZPlanes);
        velxFileStream_.write((char *) zCoorPlane_, sizeof(float) * header_.numOfZPlanes);
        
        velyFileStream_.write((char *) & header_, sizeof(header_));
        velyFileStream_.write((char *) numTrianglesPerPlane_, sizeof(int) * header_.numOfZPlanes);
        velyFileStream_.write((char *) zCoorPlane_, sizeof(float) * header_.numOfZPlanes);
        
        velzFileStream_.write((char *) & header_, sizeof(header_));
        velzFileStream_.write((char *) numTrianglesPerPlane_, sizeof(int) * header_.numOfZPlanes);
        velzFileStream_.write((char *) zCoorPlane_, sizeof(float) * header_.numOfZPlanes);
    }
    
}

bool TrifileWriter::isOpen(){
    if(! isVelocity_)
        return vertexFileStream_.is_open() && densityFileStream_.is_open();
    else{
        return vertexFileStream_.is_open() && densityFileStream_.is_open()
        && velxFileStream_.is_open() && velyFileStream_.is_open()
        && velzFileStream_.is_open();
    }
}

bool TrifileWriter::good(){
    if(! isVelocity_)
        return vertexFileStream_.good() && vertexFileStream_.good();
    else{
        return vertexFileStream_.good() && densityFileStream_.good()
        && velxFileStream_.good() && velyFileStream_.good()
        && velzFileStream_.good();
    }
}

// deprecated
void TrifileWriter::write(int * trianglesPerPlane,
                          vector<int> & trianglePlaneIds_,
                          vector<float> & vertexData_,
                          vector<float> & densityData_){
    write(trianglesPerPlane,
          & trianglePlaneIds_,
          & vertexData_,
          & densityData_);
    
    /*memset(numTrianglesPerPlaneCurrentBlock_, 0, sizeof(int) * header_.numOfZPlanes);
    header_.NumBlocks ++;
    
    outputinds = new int[trianglePlaneIds_.size()];
    int64_t numOfTrisCurrentPlane = trianglesPerPlane[0];
    numTrianglesPerPlane_[0] += trianglesPerPlane[0];
    numTrianglesPerPlaneCurrentBlock_[0] = 0;
    for(int m = 1; m < header_.numOfZPlanes; m++){
        numTrianglesPerPlaneCurrentBlock_[m] = numTrianglesPerPlaneCurrentBlock_[m-1] + trianglesPerPlane[m-1];
        numOfTrisCurrentPlane += trianglesPerPlane[m];
        numTrianglesPerPlane_[m] += trianglesPerPlane[m];
    }
    
    header_.TotalTriangles += numOfTrisCurrentPlane;
    
    for(unsigned int m = 0; m < trianglePlaneIds_.size(); m++){
        outputinds[numTrianglesPerPlaneCurrentBlock_[trianglePlaneIds_[m]]] = m;
        numTrianglesPerPlaneCurrentBlock_[trianglePlaneIds_[m]] ++;
    }
    
    TriBlockHeader blockHeader;
    blockHeader.TotalTriangles = numOfTrisCurrentPlane;
    //printf("Current Block: %d\n", numOfTrisCurrentPlane);
    vertexFileStream_.write((char *) & blockHeader, sizeof(blockHeader));
    densityFileStream_.write((char *) & blockHeader, sizeof(blockHeader));
    
    vertexFileStream_.write((char *) trianglesPerPlane, sizeof(int) * header_.numOfZPlanes);
    densityFileStream_.write((char *) trianglesPerPlane, sizeof(int) * header_.numOfZPlanes);
    
    for(unsigned int m = 0; m < trianglePlaneIds_.size(); m++){
        vertexFileStream_.write((char *) (vertexData_.data() + outputinds[m] * 6), sizeof(float) * 6);
        densityFileStream_.write((char *) (densityData_.data() + outputinds[m]), sizeof(float));
    }
    
    delete[] outputinds;
    outputinds = NULL;*/
    
}



void TrifileWriter::write(int * trianglesPerPlane,
                          std::vector<int> * trianglePlaneIds_,
                          std::vector<float> * vertexData_,
                          std::vector<float> * densityData_){
    
    
    //printf("Ok1\n");
    setTrisPerPlane(trianglesPerPlane, *trianglePlaneIds_);
    //printf("Ok2\n");
    
    TriBlockHeader blockHeader;
    blockHeader.TotalTriangles = numOfTrisCurrentPlane_;

    vertexFileStream_.write((char *) & blockHeader, sizeof(blockHeader));
    densityFileStream_.write((char *) & blockHeader, sizeof(blockHeader));
    
    vertexFileStream_.write((char *) trianglesPerPlane, sizeof(int) * header_.numOfZPlanes);
    densityFileStream_.write((char *) trianglesPerPlane, sizeof(int) * header_.numOfZPlanes);
    
    
    
    //float * densSorted = new float[densityData_->size()];
    //float * vertexSorted = new float[vertexData_->size()];
    if(cBufferSize_ < trianglePlaneIds_->size()){
        densSorted = (float *) realloc(densSorted, sizeof(float) * (densityData_->size()));
        vertexSorted = (float *) realloc(vertexSorted, sizeof(float) * (vertexData_->size()));
        
        cBufferSize_ = trianglePlaneIds_->size();
    }
    
    
    for(unsigned int m = 0; m < trianglePlaneIds_->size(); m++){
        //vertexFileStream_.write((char *) (vertexData_.data() + outputinds[m] * 6), sizeof(float) * 6);
        //densityFileStream_.write((char *) (densityData_.data() + outputinds[m]), sizeof(float));
        densSorted[m] = (*densityData_)[outputinds[m]];
        vertexSorted[m * 6 + 0] =(*vertexData_)[outputinds[m] * 6 + 0];
        vertexSorted[m * 6 + 1] =(*vertexData_)[outputinds[m] * 6 + 1];
        vertexSorted[m * 6 + 2] =(*vertexData_)[outputinds[m] * 6 + 2];
        vertexSorted[m * 6 + 3] =(*vertexData_)[outputinds[m] * 6 + 3];
        vertexSorted[m * 6 + 4] =(*vertexData_)[outputinds[m] * 6 + 4];
        vertexSorted[m * 6 + 5] =(*vertexData_)[outputinds[m] * 6 + 5];
    }
    //printf("Ok3\n");
    
    vertexFileStream_.write((char *) (vertexSorted), sizeof(float)  * (vertexData_->size()));
    densityFileStream_.write((char *) (densSorted), sizeof(float) * (densityData_->size()));
    
}



void TrifileWriter::write(int * trianglesPerPlane,
                          std::vector<int> * trianglePlaneIds_,
                          std::vector<float> * vertexData_,
                          std::vector<float> * densityData_,
                          std::vector<float> * velXData_,
                          std::vector<float> * velYData_,
                          std::vector<float> * velZData_){
    
    setTrisPerPlane(trianglesPerPlane, *trianglePlaneIds_);
    //float * densSorted = new float[densityData_->size()];
    //float * vertexSorted = new float[vertexData_->size()];
    //float * velxSorted = new float[velXData_->size()];
    //float * velySorted = new float[velYData_->size()];
    //float * velzSorted = new float[velZData_->size()];
    
    
    //printf("abb %d %d %d %d %d! \n",densityData_->size(),
    //       vertexData_->size(), velXData_->size(),
    //       velYData_->size(), velZData_->size());
    
    TriBlockHeader blockHeader;
    blockHeader.TotalTriangles = numOfTrisCurrentPlane_;
    
    vertexFileStream_.write((char *) & blockHeader, sizeof(blockHeader));
    densityFileStream_.write((char *) & blockHeader, sizeof(blockHeader));
    
    vertexFileStream_.write((char *) trianglesPerPlane, sizeof(int) * header_.numOfZPlanes);
    densityFileStream_.write((char *) trianglesPerPlane, sizeof(int) * header_.numOfZPlanes);

    
    velxFileStream_.write((char *) & blockHeader, sizeof(blockHeader));
    velyFileStream_.write((char *) & blockHeader, sizeof(blockHeader));
    velzFileStream_.write((char *) & blockHeader, sizeof(blockHeader));
    
    velxFileStream_.write((char *) trianglesPerPlane, sizeof(int) * header_.numOfZPlanes);
    velyFileStream_.write((char *) trianglesPerPlane, sizeof(int) * header_.numOfZPlanes);
    velzFileStream_.write((char *) trianglesPerPlane, sizeof(int) * header_.numOfZPlanes);
    
    
    if(cBufferSize_ < trianglePlaneIds_->size()){
        densSorted = (float *) realloc(densSorted, sizeof(float) * (densityData_->size()));
        vertexSorted = (float *) realloc(vertexSorted, sizeof(float) * (vertexData_->size()));
        
        velxSorted = (float *) realloc(velxSorted, sizeof(float) * (velXData_->size()));
        velySorted = (float *) realloc(velySorted, sizeof(float) * (velYData_->size()));
        velzSorted = (float *) realloc(velzSorted, sizeof(float) * (velZData_->size()));
        
        cBufferSize_ = trianglePlaneIds_->size();
    }
    
    

    for(unsigned int m = 0; m < trianglePlaneIds_->size(); m++){
        //vertexFileStream_.write((char *) (vertexData_.data() + outputinds[m] * 6), sizeof(float) * 6);
        //densityFileStream_.write((char *) (densityData_.data() + outputinds[m]), sizeof(float));
        densSorted[m] = (*densityData_)[outputinds[m]];
        vertexSorted[m * 6 + 0] =(*vertexData_)[outputinds[m] * 6 + 0];
        vertexSorted[m * 6 + 1] =(*vertexData_)[outputinds[m] * 6 + 1];
        vertexSorted[m * 6 + 2] =(*vertexData_)[outputinds[m] * 6 + 2];
        vertexSorted[m * 6 + 3] =(*vertexData_)[outputinds[m] * 6 + 3];
        vertexSorted[m * 6 + 4] =(*vertexData_)[outputinds[m] * 6 + 4];
        vertexSorted[m * 6 + 5] =(*vertexData_)[outputinds[m] * 6 + 5];
        
        velxSorted[m * 3 + 0] =(*velXData_)[outputinds[m] * 3 + 0];
        velxSorted[m * 3 + 1] =(*velXData_)[outputinds[m] * 3 + 1];
        velxSorted[m * 3 + 2] =(*velXData_)[outputinds[m] * 3 + 2];
        
        velySorted[m * 3 + 0] =(*velYData_)[outputinds[m] * 3 + 0];
        velySorted[m * 3 + 1] =(*velYData_)[outputinds[m] * 3 + 1];
        velySorted[m * 3 + 2] =(*velYData_)[outputinds[m] * 3 + 2];
        
        velzSorted[m * 3 + 0] =(*velZData_)[outputinds[m] * 3 + 0];
        velzSorted[m * 3 + 1] =(*velZData_)[outputinds[m] * 3 + 1];
        velzSorted[m * 3 + 2] =(*velZData_)[outputinds[m] * 3 + 2];
    }
    //printf("abb1!\n");
    vertexFileStream_.write((char *) (vertexSorted), sizeof(float) * (vertexData_->size()));
    densityFileStream_.write((char *) (densSorted), sizeof(float) * (densityData_->size()));
    velxFileStream_.write((char *) (velxSorted), sizeof(float) * (velXData_->size()));
    velyFileStream_.write((char *) (velySorted), sizeof(float) * (velYData_->size()));
    velzFileStream_.write((char *) (velzSorted), sizeof(float) * (velZData_->size()));
    
}

//use only ones before
void TrifileWriter::setTrisPerPlane(int * trianglesPerPlane,
                     std::vector<int> & trianglePlaneIds_){
    memset(numTrianglesPerPlaneCurrentBlock_, 0, sizeof(int) * header_.numOfZPlanes);
    header_.NumBlocks ++;
    
    // use for sorting the triangles for each plane
    //outputinds = new int[trianglePlaneIds_.size()];
    if(cBufferSize_ < trianglePlaneIds_.size()){
        outputinds = (int *) realloc(outputinds, sizeof(int) * trianglePlaneIds_.size());
    }
    
    
    // the number of triangles in this plane
    numOfTrisCurrentPlane_ = trianglesPerPlane[0];
    
    // add the number of triangles in current block to the total block
    numTrianglesPerPlane_[0] += trianglesPerPlane[0];
    
    // clear the first bit of this plane counter
    numTrianglesPerPlaneCurrentBlock_[0] = 0;
    
    // loop over each plane, add corresponding numbers
    for(int m = 1; m < header_.numOfZPlanes; m++){
        numTrianglesPerPlaneCurrentBlock_[m] = numTrianglesPerPlaneCurrentBlock_[m-1] + trianglesPerPlane[m-1];
        numOfTrisCurrentPlane_ += trianglesPerPlane[m];
        numTrianglesPerPlane_[m] += trianglesPerPlane[m];
    }
    
    // add the number of triangles to the total number of triangles in the whole file
    header_.TotalTriangles += numOfTrisCurrentPlane_;
    
    
    for(unsigned int m = 0; m < trianglePlaneIds_.size(); m++){
        outputinds[numTrianglesPerPlaneCurrentBlock_[trianglePlaneIds_[m]]] = m;
        numTrianglesPerPlaneCurrentBlock_[trianglePlaneIds_[m]] ++;
    }
    
}



void TrifileWriter::close(){
    //printf("%d %d\n", header_.NumBlocks, header_.numOfZPlanes);
    
    vertexFileStream_.seekp(0, ios_base::beg);
    densityFileStream_.seekp(0, ios_base::beg);
    
    
    header_.fileType = POS;
    vertexFileStream_.write((char *) & header_, sizeof(header_));
    
    header_.fileType = DENS;
    densityFileStream_.write((char *) & header_, sizeof(header_));
    

    
    /*printf("ok, %d %d %d %d %d %d\n", sizeof(TriHeader), header_.numOfZPlanes, numTrianglesPerPlane_[0], numTrianglesPerPlane_[1], numTrianglesPerPlane_[2], numTrianglesPerPlane_[3]);
    */
    
    //test
    /*for(int i = 0; i < header_.numOfZPlanes; i++){
        printf("Tris: %d\n", numTrianglesPerPlane_[i]);
    }*/
    
    vertexFileStream_.write((char *) numTrianglesPerPlane_,
                            sizeof(int) * header_.numOfZPlanes);
    densityFileStream_.write((char *) numTrianglesPerPlane_,
                             sizeof(int) * header_.numOfZPlanes);
    
    vertexFileStream_.close();
    densityFileStream_.close();
    
    
    
    
    if(isVelocity_){
        header_.fileType = VELX;
        velxFileStream_.seekp(0, ios_base::beg);
        velxFileStream_.write((char *) & header_, sizeof(header_));
        velxFileStream_.write((char *) numTrianglesPerPlane_,
                                sizeof(int) * header_.numOfZPlanes);
        velxFileStream_.close();
        
        header_.fileType = VELY;
        velyFileStream_.seekp(0, ios_base::beg);
        velyFileStream_.write((char *) & header_, sizeof(header_));
        velyFileStream_.write((char *) numTrianglesPerPlane_,
                              sizeof(int) * header_.numOfZPlanes);
        velyFileStream_.close();
        
        
        header_.fileType = VELZ;
        velzFileStream_.seekp(0, ios_base::beg);
        velzFileStream_.write((char *) & header_, sizeof(header_));
        velzFileStream_.write((char *) numTrianglesPerPlane_,
                              sizeof(int) * header_.numOfZPlanes);
        velzFileStream_.close();
    }
}









///////////reader///////////////////
TrifileReader::TrifileReader(std::string basename, bool isVelocity){
    basename_ = basename;
    isVelocity_ = isVelocity;
    
    string vertfn = basename + "." TRIFILESUFFIX "." VERTEXFILESUFFIX;
    string densfn = basename + "." TRIFILESUFFIX "." DENSITYFILESUFFIX;
    string velxfn = basename + "." TRIFILESUFFIX "." VELXFILESUFFIX;
    string velyfn = basename + "." TRIFILESUFFIX "." VELYFILESUFFIX;
    string velzfn = basename + "." TRIFILESUFFIX "." VELZFILESUFFIX;
    
    vertexFileStream_.open(vertfn.c_str(), ios::binary | ios::in);
    densityFileStream_.open(densfn.c_str(), ios::binary | ios::in);
    TriHeader header0_, headervx, headervy, headervz;
    vertexFileStream_.read((char *)&header_, sizeof(header_));
    densityFileStream_.read((char *) &header0_, sizeof(header0_));
    
    
    //printf("Ok11 : %d %d\n", header0_.TotalTriangles, header_.TotalTriangles);
    if((header0_.NumBlocks != header_.NumBlocks) || (header_.NumBlocks == 0)){
        fprintf(stderr, "Input File Incorrect!\n");
        exit(1);
    }
    
    if(isVelocity){
        velxFileStream_.open(velxfn.c_str(), ios::binary | ios::in);
        velyFileStream_.open(velyfn.c_str(), ios::binary | ios::in);
        velzFileStream_.open(velzfn.c_str(), ios::binary | ios::in);
     
        velxFileStream_.read((char *) &headervx, sizeof(headervx));
        if((headervx.NumBlocks != header_.NumBlocks)){
            fprintf(stderr, "Input Velocity X File Incorrect!\n");
            exit(1);
        }
        
        velyFileStream_.read((char *) &headervy, sizeof(headervy));
        if((headervy.NumBlocks != header_.NumBlocks)){
            fprintf(stderr, "Input Velocity Y File Incorrect!\n");
            exit(1);
        }
        
        
        velzFileStream_.read((char *) &headervz, sizeof(headervz));
        if((headervz.NumBlocks != header_.NumBlocks)){
            fprintf(stderr, "Input Velocity Z File Incorrect!\n");
            exit(1);
        }
    }
    
    

    
    //printf("%d %d\n", header_.NumBlocks, header_.numOfZPlanes);
    numTrianglesPerPlane_ = new int[header_.numOfZPlanes];
    numTrianglesPerPlaneCurrent_ = new int[header_.numOfZPlanes];
    zCoorPlane_ = new float[header_.numOfZPlanes];
    numBlocks_ = header_.NumBlocks;
    
    memset(numTrianglesPerPlane_,  0, sizeof(int) * header_.numOfZPlanes);
    memset(numTrianglesPerPlaneCurrent_,  0, sizeof(int) * header_.numOfZPlanes);
    memset(zCoorPlane_,  0, sizeof(float) * header_.numOfZPlanes);

    if(vertexFileStream_.good()){
        vertexFileStream_.read((char *) numTrianglesPerPlane_, sizeof(int) * header_.numOfZPlanes);
        vertexFileStream_.read((char *) zCoorPlane_, sizeof(float) * header_.numOfZPlanes);
    }else{
        fprintf(stderr, "Input File Incorrect: %s!\n", vertfn.c_str());
        exit(1);
    }
    
    if(densityFileStream_.good()){
        densityFileStream_.read((char *) numTrianglesPerPlane_, sizeof(int) * header_.numOfZPlanes);
        densityFileStream_.read((char *) zCoorPlane_, sizeof(float) * header_.numOfZPlanes);
    }else{
        fprintf(stderr, "Input File Incorrect: %s!\n", vertfn.c_str());
        exit(1);
    }
    
    
    
    //printf("ok, %d %d %d %d %d \n", sizeof(TriHeader), numTrianglesPerPlane_[0], numTrianglesPerPlane_[1], numTrianglesPerPlane_[2], numTrianglesPerPlane_[3]);
    //test
    //for(int i = 0; i < header_.numOfZPlanes; i++){
    //    printf("Tris: %d\n", numTrianglesPerPlane_[i]);
    //}
    
    
    //test:
    TriBlockHeader head0, head1;
    vertexFileStream_.read((char *) & head0, sizeof(head0));
    densityFileStream_.read((char *) & head1, sizeof(head1));
    
    //printf("Ok--: %f %f\n", zCoorPlane_[63], zCoorPlane_[63]);
    //printf("Ok++ : %d %d\n", head0.TotalTriangles, head1.TotalTriangles);
    
    int maxNumTris = *std::max_element(numTrianglesPerPlane_, numTrianglesPerPlane_+header_.numOfZPlanes);
    
    vertexData_ = new float[maxNumTris * 6];
    densityData_ = new float[maxNumTris];
    
    if(isVelocity_){
        velXData_ = new float[maxNumTris * 3];
        velYData_ = new float[maxNumTris * 3];
        velZData_ = new float[maxNumTris * 3];
    }
    
    //test
    //exit(0);

}

void TrifileReader::close(){
    vertexFileStream_.close();
    densityFileStream_.close();
    if(isVelocity_){
        velxFileStream_.close();
        velyFileStream_.close();
        velzFileStream_.close();
    }
    
}


bool TrifileReader::isOpen(){
    if(! isVelocity_){
        return vertexFileStream_.is_open() && densityFileStream_.is_open();
    }
    else{
        return vertexFileStream_.is_open() && densityFileStream_.is_open()
        && velxFileStream_.is_open() && velyFileStream_.is_open()
        && velzFileStream_.is_open();
    }
}

float TrifileReader::getZcoor(int plane){
    return zCoorPlane_[plane];
}
float TrifileReader::getNumTriangles(int plane){
    return numTrianglesPerPlane_[plane];
}

void TrifileReader::loadPlane(int plane){
    //printf("Ok: %d\n", plane);
    
    int64_t densitypos = 0 + sizeof(header_) + sizeof(int) * (int64_t) header_.numOfZPlanes
        + sizeof(float) * header_.numOfZPlanes;
    
    int64_t vertexpos = 0 + sizeof(header_) + sizeof(int) * (int64_t) header_.numOfZPlanes
        + sizeof(float) * header_.numOfZPlanes;
    
    int64_t velpos = 0 + sizeof(header_) + sizeof(int) * (int64_t) header_.numOfZPlanes
        + sizeof(float) * header_.numOfZPlanes;
    
    vertexFileStream_.clear();
    densityFileStream_.clear();
    if(isVelocity_){
        velxFileStream_.clear();
        velyFileStream_.clear();
        velzFileStream_.clear();
    }
    
    int64_t dataposDens = 0;
    int64_t dataposVert = 0;
    int64_t dataposVel = 0;
    
    TriBlockHeader head0, head1;
    TriBlockHeader headvx, headvy, headvz;
    
    //printf("Ok Blocks: %d %d %d \n", header_.NumBlocks, densitypos, vertexpos);
    for(int i = 0; i < header_.NumBlocks; i++){
        //printf("Ok n: %d\n", i);
        vertexFileStream_.seekg(vertexpos, ios_base::beg);
        densityFileStream_.seekg(densitypos, ios_base::beg);
        
        vertexFileStream_.read((char *) & head0, sizeof(head0));
        densityFileStream_.read((char *) & head1, sizeof(head1));
        
        //printf("Ok : %d %d\n", head0.TotalTriangles, head1.TotalTriangles);
        if(head0.TotalTriangles != head1.TotalTriangles){
            fprintf(stderr, "Input File Density/Pos Incorrect!\n");
            exit(1);
        }
        
        
        densityFileStream_.read((char *) numTrianglesPerPlaneCurrent_,
                                sizeof(int) * header_.numOfZPlanes);
        vertexFileStream_.read((char *) numTrianglesPerPlaneCurrent_,
                               sizeof(int) * header_.numOfZPlanes);
        
        
        
        
        densitypos = densityFileStream_.tellg();
        vertexpos = vertexFileStream_.tellg();
        
        if(isVelocity_){
            velxFileStream_.seekg(velpos, ios_base::beg);
            velyFileStream_.seekg(velpos, ios_base::beg);
            velzFileStream_.seekg(velpos, ios_base::beg);
            
            velxFileStream_.read((char *) & headvx, sizeof(headvx));
            velxFileStream_.read((char *) numTrianglesPerPlaneCurrent_,
                                   sizeof(int) * header_.numOfZPlanes);
            
            //printf("VX: %d\n", headvx.TotalTriangles);
            if(head0.TotalTriangles != headvx.TotalTriangles){
                fprintf(stderr, "Input Velx File Incorrect!\n");
                exit(1);
            }
            
            
            velyFileStream_.read((char *) & headvy, sizeof(headvy));
            velyFileStream_.read((char *) numTrianglesPerPlaneCurrent_,
                                 sizeof(int) * header_.numOfZPlanes);
            //printf("VY: %d\n", headvy.TotalTriangles);
            if(head0.TotalTriangles != headvy.TotalTriangles){
                fprintf(stderr, "Input Vely File Incorrect!\n");
                exit(1);
            }
            
            
            velzFileStream_.read((char *) & headvz, sizeof(headvz));
            velzFileStream_.read((char *) numTrianglesPerPlaneCurrent_,
                                 sizeof(int) * header_.numOfZPlanes);
            //printf("VZ: %d\n", headvz.TotalTriangles);
            if(head0.TotalTriangles != headvz.TotalTriangles){
                fprintf(stderr, "Input Velz File Incorrect!\n");
                exit(1);
            }
            
            velpos = velxFileStream_.tellg();
        }
        
        

        //densitypos += sizeof(int) * header_.numOfZPlanes;
        //vertexpos += sizeof(int) * header_.numOfZPlanes;
        

        //test
        /*for(int ll = 0; ll < header_.numOfZPlanes; ll++){
            printf("CCCC %d\n", numTrianglesPerPlaneCurrent_[ll]);
        }*/

        
        //vertexFileStream_.seekg(vertexpos, ios_base::beg);
        //densityFileStream_.seekg(densitypos, ios_base::beg);
        

        
        int64_t densitypos0 = densitypos;
        int64_t vertexpos0 = vertexpos;
        int64_t velpos0 = velpos;
        
        for(int j = 0; j < plane; j++){
            densitypos += numTrianglesPerPlaneCurrent_[j] * sizeof(float);
            vertexpos += numTrianglesPerPlaneCurrent_[j] * sizeof(float) * 6;
            velpos += numTrianglesPerPlaneCurrent_[j] * sizeof(float) * 3;
        }
        
        vertexFileStream_.seekg(vertexpos, ios_base::beg);
        densityFileStream_.seekg(densitypos, ios_base::beg);
        vertexFileStream_.read((char *) (vertexData_+dataposVert), sizeof(float) * numTrianglesPerPlaneCurrent_[plane] * 6);

        densityFileStream_.read((char *) (densityData_+dataposDens), sizeof(float) * numTrianglesPerPlaneCurrent_[plane]);
        
        if(isVelocity_){
            velxFileStream_.seekg(velpos, ios_base::beg);
            velyFileStream_.seekg(velpos, ios_base::beg);
            velzFileStream_.seekg(velpos, ios_base::beg);
            velxFileStream_.read((char *) (velXData_+dataposVel),
                                 sizeof(float) * numTrianglesPerPlaneCurrent_[plane] * 3);
            velyFileStream_.read((char *) (velYData_+dataposVel),
                                 sizeof(float) * numTrianglesPerPlaneCurrent_[plane] * 3);
            velzFileStream_.read((char *) (velZData_+dataposVel),
                                 sizeof(float) * numTrianglesPerPlaneCurrent_[plane] * 3);
            
        }

        densitypos = densitypos0 + sizeof(float) * head0.TotalTriangles;
        vertexpos = vertexpos0 + sizeof(float) * head1.TotalTriangles * 6;
        velpos = velpos0 + sizeof(float) * headvx.TotalTriangles * 3;
        
        //printf("Ok end: %d\n", i);
        
        //test
        /*if(plane == 0){
            for(int m = 0; m < numTrianglesPerPlaneCurrent_[plane]; m++){
                printf("Vert: %d %f %f %f %f %f %f\n", m,
                       vertexData_[dataposVert + m * 6 + 0],
                       vertexData_[dataposVert + m * 6 + 1],
                       vertexData_[dataposVert + m * 6 + 2],
                       vertexData_[dataposVert + m * 6 + 3],
                       vertexData_[dataposVert + m * 6 + 4],
                       vertexData_[dataposVert + m * 6 + 5]);
                printf("Dens: %e\n", densityData_[dataposDens + m]);
            }
         
        }*/
        
        dataposVert += numTrianglesPerPlaneCurrent_[plane] * 6;
        dataposDens += numTrianglesPerPlaneCurrent_[plane];
        dataposVel += numTrianglesPerPlaneCurrent_[plane] * 3;

    }
    
    
}
float * TrifileReader::getDensity(){
    return densityData_;
}

float * TrifileReader::getTriangles(){
    return vertexData_;
}


float * TrifileReader::getVelocityX(){
    return velXData_;
}

float * TrifileReader::getVelocityY(){
    return velYData_;
}

float * TrifileReader::getVelocityZ(){
    return velZData_;
}


TrifileReader::~TrifileReader(){
    delete[] numTrianglesPerPlane_;
    delete[] numTrianglesPerPlaneCurrent_;
    delete[] zCoorPlane_;
    delete[] vertexData_;
    delete[] densityData_;
    
    if(isVelocity_){
        delete[] velXData_;
        delete[] velYData_;
        delete[] velZData_;
    }
}


