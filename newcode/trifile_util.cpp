#include <cstring>
#include <vector>
#include <algorithm>
#include "trifile_util.h"


using namespace std;

TrifileWriter::TrifileWriter(TriHeader header){
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
}

TrifileWriter::~TrifileWriter(){
    close();
    
    delete[] numTrianglesPerPlane_;
    delete[] zCoorPlane_;
    delete[] numTrianglesPerPlaneCurrentBlock_;
}


//open the file and prepare the file for writing
void TrifileWriter::open(std::string basename){
    
    
    string vertfn = basename + "."TRIFILESUFFIX"."VERTEXFILESUFFIX;
    string densfn = basename + "."TRIFILESUFFIX"."DENSITYFILESUFFIX;
    vertexFileStream_.open(vertfn.c_str(), ios::binary | ios::out);
    densityFileStream_.open(densfn.c_str(), ios::binary | ios::out);
    vertexFileStream_.write((char *) & header_, sizeof(header_));
    densityFileStream_.write((char *) & header_, sizeof(header_));
    
    /*printf("init ok, %d %d %d %d %d \n", header_.numOfZPlanes, numTrianglesPerPlane_[0], numTrianglesPerPlane_[1], numTrianglesPerPlane_[2], numTrianglesPerPlane_[3]);*/
    
    vertexFileStream_.write((char *) numTrianglesPerPlane_, sizeof(int) * header_.numOfZPlanes);
    vertexFileStream_.write((char *) zCoorPlane_, sizeof(float) * header_.numOfZPlanes);
    
    
    densityFileStream_.write((char *) numTrianglesPerPlane_, sizeof(int) * header_.numOfZPlanes);
    densityFileStream_.write((char *) zCoorPlane_, sizeof(float) * header_.numOfZPlanes);
    
}

bool TrifileWriter::isOpen(){
    return vertexFileStream_.is_open() && densityFileStream_.is_open();
}

bool TrifileWriter::good(){
    return vertexFileStream_.good() && vertexFileStream_.good();
}

void TrifileWriter::write(int * trianglesPerPlane,
                          vector<int> & trianglePlaneIds_,
                          vector<float> & vertexData_,
                          vector<float> & densityData_){
    
    memset(numTrianglesPerPlaneCurrentBlock_, 0, sizeof(int) * header_.numOfZPlanes);
    header_.NumBlocks ++;
    
    int * outputinds = new int[trianglePlaneIds_.size()];
    int numOfTrisCurrentPlane = trianglesPerPlane[0];
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
        /*if(trianglePlaneIds_[outputinds[m]] == 0){
            printf("Vert: %d %f %f %f %f %f %f\n", m, vertexData_[outputinds[m] * 6 + 0],
               vertexData_[outputinds[m] * 6 + 1],
               vertexData_[outputinds[m] * 6 + 2],
               vertexData_[outputinds[m] * 6 + 3],
               vertexData_[outputinds[m] * 6 + 4],
               vertexData_[outputinds[m] * 6 + 5]);
            printf("Dens: %e\n", densityData_[outputinds[m]]);
        }*/
    }
    
}

void TrifileWriter::close(){
    //printf("%d %d\n", header_.NumBlocks, header_.numOfZPlanes);
    
    vertexFileStream_.seekp(0, ios_base::beg);
    densityFileStream_.seekp(0, ios_base::beg);
    
    vertexFileStream_.write((char *) & header_, sizeof(header_));
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
}





///////////reader///////////////////
TrifileReader::TrifileReader(std::string basename){
    basename_ = basename;
    string vertfn = basename + "."TRIFILESUFFIX"."VERTEXFILESUFFIX;
    string densfn = basename + "."TRIFILESUFFIX"."DENSITYFILESUFFIX;
    vertexFileStream_.open(vertfn.c_str(), ios::binary | ios::in);
    densityFileStream_.open(densfn.c_str(), ios::binary | ios::in);
    vertexFileStream_.read((char *)&header_, sizeof(header_));
    
    TriHeader header0_;
    densityFileStream_.read((char *) &header0_, sizeof(header0_));
    
    if((header0_.NumBlocks != header_.NumBlocks) || (header_.NumBlocks == 0)){
        fprintf(stderr, "Input File Incorrect!\n");
        exit(1);
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
    /*for(int i = 0; i < header_.numOfZPlanes; i++){
        printf("Tris: %d\n", numTrianglesPerPlane_[i]);
    }*/
    
    int maxNumTris = *std::max_element(numTrianglesPerPlane_, numTrianglesPerPlane_+header_.numOfZPlanes);
    
    

    
    vertexData_ = new float[maxNumTris * 6];
    densityData_ = new float[maxNumTris];
    
    //test
    //exit(0);

}

void TrifileReader::close(){
    vertexFileStream_.close();
    densityFileStream_.close();
}


bool TrifileReader::isOpen(){
    return vertexFileStream_.is_open() && densityFileStream_.is_open();
}

float TrifileReader::getZcoor(int plane){
    return zCoorPlane_[plane];
}
float TrifileReader::getNumTriangles(int plane){
    return numTrianglesPerPlane_[plane];
}

void TrifileReader::loadPlane(int plane){
    //printf("Ok: %d\n", plane);
    
    int densitypos = 0 + sizeof(header_) + sizeof(int) * header_.numOfZPlanes
        + sizeof(float) * header_.numOfZPlanes;
    int vertexpos = 0 + sizeof(header_) + sizeof(int) * header_.numOfZPlanes
        + sizeof(float) * header_.numOfZPlanes;
    
    vertexFileStream_.clear();
    densityFileStream_.clear();
    int dataposDens = 0;
    int dataposVert = 0;
    
    TriBlockHeader head0, head1;
    
    //printf("Ok Blocks: %d %d %d \n", header_.NumBlocks, densitypos, vertexpos);
    for(int i = 0; i < header_.NumBlocks; i++){
        //printf("Ok n: %d\n", i);
        vertexFileStream_.seekg(vertexpos, ios_base::beg);
        densityFileStream_.seekg(densitypos, ios_base::beg);
        
        vertexFileStream_.read((char *) & head0, sizeof(head0));
        densityFileStream_.read((char *) & head1, sizeof(head1));
        
        //printf("Ok : %d %d\n", head0.TotalTriangles, head1.TotalTriangles);
        if(head0.TotalTriangles != head1.TotalTriangles){
            fprintf(stderr, "Input File Incorrect!\n");
            exit(1);
        }
        
        

        //densitypos += sizeof(int) * header_.numOfZPlanes;
        //vertexpos += sizeof(int) * header_.numOfZPlanes;
        
        densityFileStream_.read((char *) numTrianglesPerPlaneCurrent_,
                                sizeof(int) * header_.numOfZPlanes);
        vertexFileStream_.read((char *) numTrianglesPerPlaneCurrent_,
                                sizeof(int) * header_.numOfZPlanes);
        
        //test
        /*for(int ll = 0; ll < header_.numOfZPlanes; ll++){
            printf("CCCC %d\n", numTrianglesPerPlaneCurrent_[ll]);
        }*/

        
        //vertexFileStream_.seekg(vertexpos, ios_base::beg);
        //densityFileStream_.seekg(densitypos, ios_base::beg);
        
        
        
        densitypos = densityFileStream_.tellg();
        vertexpos = vertexFileStream_.tellg();
        
        int densitypos0 = densitypos;
        int vertexpos0 = vertexpos;
        for(int j = 0; j < plane; j++){
            densitypos += numTrianglesPerPlaneCurrent_[j] * sizeof(float);
            vertexpos += numTrianglesPerPlaneCurrent_[j] * sizeof(float) * 6;
        }
        
        vertexFileStream_.seekg(vertexpos, ios_base::beg);
        densityFileStream_.seekg(densitypos, ios_base::beg);
        vertexFileStream_.read((char *) (vertexData_+dataposVert), sizeof(float) * numTrianglesPerPlaneCurrent_[plane] * 6);

        densityFileStream_.read((char *) (densityData_+dataposDens), sizeof(float) * numTrianglesPerPlaneCurrent_[plane]);

        densitypos = densitypos0 + sizeof(float) * head0.TotalTriangles;
        vertexpos = vertexpos0 + sizeof(float) * head1.TotalTriangles * 6;
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

    }
    
    
}
float * TrifileReader::getDensity(int plane){
    return densityData_;
}

float * TrifileReader::getTriangles(int plane){
    return vertexData_;
}

TrifileReader::~TrifileReader(){
    delete[] numTrianglesPerPlane_;
    delete[] numTrianglesPerPlaneCurrent_;
    delete[] zCoorPlane_;
    delete[] vertexData_;
    delete[] densityData_;
}


