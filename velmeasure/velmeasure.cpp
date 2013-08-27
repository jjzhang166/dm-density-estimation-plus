//#define __OMP__

#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <stdint.h>
#include <vector>
#include <map>
#include <set>
#include <cfloat>

#ifdef __OMP__
#include <omp.h>
#endif

#include <kdtree++/kdtree.hpp>

#include "readgadget.h"

#define FIND_EPSILON FLT_EPSILON
#define OUTPUTBUFFER 65536
#define OUTPUTLINELENGTH 6


//#define TREE_CODE

using namespace std;

#define SQRT3 1.73205080757

//using namespace std;

string singlefilename = "";
string prefix = "";
string basename_ = "";
int numOfFiles = 0;
string outputfilename = "";
uint64_t numOfPairs = 0;

Point * allpos;
Point * allvel;

float outputbuffer[OUTPUTBUFFER * OUTPUTLINELENGTH];


struct kdtreeNode
{
    typedef REAL value_type;
    
    size_t index;
    
    value_type operator[](size_t n) const
    {
        switch (n) {
            case 0:
                return allpos[index].x;
            case 1:
                return allpos[index].y;
            case 2:
                return allpos[index].z;
            default:
                return allpos[index].x;
        }
        return allpos[index].x;
    }
    
};

typedef KDTree::KDTree<3,kdtreeNode> treeType;
treeType kdtree;


vector<kdtreeNode> retVec;

//find the particles in between 2 cube shells
//r1 and r2 are the 1/2 of the edge length of the 2 cube
//point p is the center of the 2 cubes
//r1 < r2
void findPartsInShell(treeType &tree, Point & p,
                      REAL r1, REAL r2,
                      vector<kdtreeNode> & retVec){
    kdtreeNode tempnod;
    tempnod.index = 0;
    
    treeType::_Region_ * region = new treeType::_Region_(tempnod);
    
    //up
    region->_M_low_bounds[0] = p.x - r2;
    region->_M_high_bounds[0] = p.x + r2 + FIND_EPSILON;
    
    region->_M_low_bounds[1] = p.y - r2;
    region->_M_high_bounds[1] = p.y + r2 + FIND_EPSILON;
    
    region->_M_low_bounds[2] = p.z + r1;
    region->_M_high_bounds[2] = p.z + r2 + FIND_EPSILON;
    
    tree.find_within_range(*region,
                           back_insert_iterator<vector<kdtreeNode> >(retVec));
    
    //down
    region->_M_low_bounds[0] = p.x - r2;
    region->_M_high_bounds[0] = p.x + r2 + FIND_EPSILON;
    
    region->_M_low_bounds[1] = p.y - r2;
    region->_M_high_bounds[1] = p.y + r2 + FIND_EPSILON;
    
    region->_M_low_bounds[2] = p.z - r2;
    region->_M_high_bounds[2] = p.z - r1 + FIND_EPSILON;
    
    tree.find_within_range(*region,
                           back_insert_iterator<vector<kdtreeNode> >(retVec));
    
    //left
    region->_M_low_bounds[0] = p.x - r2;
    region->_M_high_bounds[0] = p.x - r1 + FIND_EPSILON;
    
    region->_M_low_bounds[1] = p.y - r2;
    region->_M_high_bounds[1] = p.y + r2 + FIND_EPSILON;
    
    region->_M_low_bounds[2] = p.z - r1;
    region->_M_high_bounds[2] = p.z + r1 + FIND_EPSILON;
    
    tree.find_within_range(*region,
                           back_insert_iterator<vector<kdtreeNode> >(retVec));
    
    
    //right
    region->_M_low_bounds[0] = p.x + r1;
    region->_M_high_bounds[0] = p.x + r2 + FIND_EPSILON;
    
    region->_M_low_bounds[1] = p.y - r2;
    region->_M_high_bounds[1] = p.y + r2 + FIND_EPSILON;
    
    region->_M_low_bounds[2] = p.z - r1;
    region->_M_high_bounds[2] = p.z + r1 + FIND_EPSILON;
    
    tree.find_within_range(*region,
                           back_insert_iterator<vector<kdtreeNode> >(retVec));
    

    //front
    region->_M_low_bounds[0] = p.x - r1;
    region->_M_high_bounds[0] = p.x + r1 + FIND_EPSILON;
    
    region->_M_low_bounds[1] = p.y  - r2;
    region->_M_high_bounds[1] = p.y - r1 + FIND_EPSILON;
    
    region->_M_low_bounds[2] = p.z - r1;
    region->_M_high_bounds[2] = p.z + r1 + FIND_EPSILON;
    
    tree.find_within_range(*region,
                           back_insert_iterator<vector<kdtreeNode> >(retVec));
    
    
    //back
    region->_M_low_bounds[0] = p.x - r1;
    region->_M_high_bounds[0] = p.x + r1 + FIND_EPSILON;
    
    region->_M_low_bounds[1] = p.y  + r1;
    region->_M_high_bounds[1] = p.y + r2 + FIND_EPSILON;
    
    region->_M_low_bounds[2] = p.z - r1;
    region->_M_high_bounds[2] = p.z + r1 + FIND_EPSILON;
    
    tree.find_within_range(*region,
                           back_insert_iterator<vector<kdtreeNode> >(retVec));
    

}

void printUsage(string pname){
    fprintf(stderr, "Usage: %s <Single Filename> <outputfilename> <{ radius list }> <{ shellsize list }>\n",
           pname.c_str()
           );
    fprintf(stderr, "-OR-\n");
    fprintf(stderr, "%s <prefix> <basename> <numOfFiles> <outputfilename> <{ radius list }> <{ shellsize list }>\n",
           pname.c_str()
           );
    fprintf(stderr, "There must be SPACES before and after \"{\"\n");
    fprintf(stderr, "The list size of radius and shellsize must be equal. \n");
    fprintf(stderr, "Data will be outputed in binary format (pvm):\n");
    fprintf(stderr, "  [Num of Pairs]: uint64_t\n");
    fprintf(stderr, "  [rx]: float32, code units (kpc/n or mpc/h)\n");
    fprintf(stderr, "  [ry]: float32, code units (kpc/n or mpc/h)\n");
    fprintf(stderr, "  [rz]: float32, code units (kpc/n or mpc/h)\n");
    fprintf(stderr, "  [vx]: float32, km/s\n");
    fprintf(stderr, "  [vy]: float32, km/s\n");
    fprintf(stderr, "  [vz]: float32, km/s\n");
}




int main(int argv, char * args[]){
    GSnap * psnap;
    
    vector<REAL> radiusList;
    vector<REAL> shellSizeList;
    
    int k = 0;
    if(argv < 4){
        printUsage(args[0]);
        exit(1);
    }else if(strcmp(args[3], "{") == 0){
        singlefilename = args[1];
        outputfilename = args[2];
        
        fprintf(stderr, "Fileame: %s\n", singlefilename.c_str());
        fprintf(stderr, "OutputFile: %s\n", outputfilename.c_str());
        k = 4;
        
    }else if(argv >= 6 && strcmp(args[5], "{") == 0){
        prefix = args[1];
        basename_ = args[2];
        numOfFiles = atoi(args[3]);

        fprintf(stderr, "Prefix: %s\n", prefix.c_str());
        fprintf(stderr, "BaseName: %s\n", basename_.c_str());
        fprintf(stderr, "Num of Files: %d\n", numOfFiles);
        k = 6;
    }else{
        printUsage(args[0]);
        exit(1);
    }
    
    while(strcmp(args[k], "}") != 0){
        stringstream ss;
        REAL radius;
        ss << args[k];
        ss >> radius;
        radiusList.push_back(radius);
        k++;
    }
    k++;
    if(strcmp(args[k], "{") != 0){
        printUsage(args[0]);
        exit(1);
    }
    k++;
    while(strcmp(args[k], "}") != 0){
        stringstream ss;
        REAL shell;
        ss << args[k];
        ss >> shell;
        shellSizeList.push_back(shell);
        k++;
    }
    if(shellSizeList.size() != radiusList.size()){
        fprintf(stderr, "Radius list does shell size\n");
        printUsage(args[0]);
        exit(1);
    }

    
    fprintf(stderr, "radius =");
    for(size_t i = 0; i < radiusList.size(); i++){
        fprintf(stderr, " %f", radiusList[i]);
    }
    fprintf(stderr, "\nshellsize = ");
    for(size_t i = 0; i < shellSizeList.size(); i++){
        fprintf(stderr, " %f", shellSizeList[i]);
    }
    fprintf(stderr, "\n");
    
    //test
    
    //return 0;
    
    
    if(numOfFiles == 0 && singlefilename == ""){
        printUsage(args[0]);
        exit(1);
    }
    

    if(numOfFiles != 0){
        //fprintf(stderr, "Create file!\n");
        psnap = new GSnap(
                        prefix, 
                        basename_, 
                        numOfFiles, 
                        true, 
                        1, 
                        -1);
    }else{
        psnap = new GSnap(singlefilename, true, 1, -1);
    }
    
    allpos = psnap -> getAllPos();
    allvel = psnap -> getAllVel();
    size_t nparts = psnap -> Npart;
    retVec.reserve(nparts);
    
    //a = 1 / (z + 1)
    double redshift = psnap->header.redshift;
    double a = 1.0 / ( redshift + 1.0);
    double sqa = sqrt(a);
    
    //convert velocity to km/s (peculiar velocity)
    for(size_t i = 0; i < nparts; i++){
        allvel[i] = allvel[i] * sqa;
    }
    
    // make random 3d points
    size_t m_count = nparts / 50;

    fprintf(stderr, "BoxSize = %f\n", psnap->header.BoxSize);
    fprintf(stderr, "a = %f\n", a);
    fprintf(stderr, "Building tree ...\n");
    for ( size_t n = 0; n < nparts; ++n)
    {
        
        kdtreeNode node;
        node.index = n;
        kdtree.insert( node);
        if(n % m_count == 0){
            fprintf(stderr, ">");
            cout.flush();
        }
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "Optimising tree ...\n");
    kdtree.optimise();
    fprintf(stderr, "KDTree Built. Nparts: %ld\n", nparts);
    
    fstream outputStream(outputfilename.c_str(), ios::out | ios::binary);
    if(!outputStream.good()){
        fprintf(stderr, "Cannot OPEN/CREATE output file: %s\n", outputfilename.c_str());
    }

    uint64_t temp_numpairs = 0;
    outputStream.write((char *) &numOfPairs, sizeof(uint64_t));
    for(size_t _idr = 0; _idr < radiusList.size(); _idr ++){
        float radius = radiusList[_idr];
        float shellsize = shellSizeList[_idr];
        fprintf(stderr, "Radius: %f, shell %f\n", radius, shellsize);
        for( size_t i = 0; i < nparts; i ++){
            if(i % m_count == 0){
                fprintf(stderr, ">");
                cout.flush();
            }
            retVec.clear();
            findPartsInShell(kdtree, allpos[i],
                             radius / SQRT3 / 1.01, radius + shellsize * 1.01,
                             retVec);
            for(size_t j = 0; j < retVec.size(); j ++){
                //printf("ok1 %d\n", i);
                Point rvector = allpos[retVec[j].index] - allpos[i];
                //printf("ok2 %d\n", j);
                float r = sqrt(rvector.dot(rvector));
                //rvector = rvector / r;
                if(r >= radius && r < radius + shellsize){
                    Point wvector = allvel[retVec[j].index] - allvel[i];
                    outputbuffer[temp_numpairs * OUTPUTLINELENGTH + 0] = rvector.x;
                    outputbuffer[temp_numpairs * OUTPUTLINELENGTH + 1] = rvector.y;
                    outputbuffer[temp_numpairs * OUTPUTLINELENGTH + 2] = rvector.z;
                    outputbuffer[temp_numpairs * OUTPUTLINELENGTH + 3] = wvector.x;
                    outputbuffer[temp_numpairs * OUTPUTLINELENGTH + 4] = wvector.y;
                    outputbuffer[temp_numpairs * OUTPUTLINELENGTH + 5] = wvector.z;
                    temp_numpairs ++;
                    numOfPairs ++;
                    if(temp_numpairs >= OUTPUTBUFFER){
                        //output
                        outputStream.write((char *) outputbuffer, sizeof(float) * temp_numpairs * OUTPUTLINELENGTH);
                        temp_numpairs = 0;
                    }
                }
            }
            
        }
        fprintf(stderr, "\n");

    }
    outputStream.write((char *) outputbuffer, sizeof(float) * temp_numpairs * OUTPUTLINELENGTH);
    temp_numpairs = 0;
    outputStream.seekg(0, outputStream.beg);
    outputStream.write((char *) &numOfPairs, sizeof(uint64_t));
    outputStream.close();
    fprintf(stderr, "Finished, find %lld pairs.\n", numOfPairs);
    
    delete psnap;
}
