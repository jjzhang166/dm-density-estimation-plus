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

#include <kdtree++/kdtree.hpp>

#include "readgadget.h"


#define TREE_CODE

#define NUMBINS 2000
#define BINSIZE 1.0
#define MAX_W 1000.0
#define SQRT2 1.41421356237

using namespace std;

string singlefilename = "";
string prefix = "";
string basename = "";
int numOfFiles = 0;
float radius = 10000.0;
float shellsize = 1000.0;
Point wvector;
Point rvector;
uint32_t z_w[NUMBINS];


Point * allpos;
Point * allvel;


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
    region->_M_high_bounds[0] = p.x + r2;
    
    region->_M_low_bounds[1] = p.y - r2;
    region->_M_high_bounds[1] = p.y + r2;
    
    region->_M_low_bounds[2] = p.z + r1;
    region->_M_high_bounds[2] = p.z + r2;
    
    tree.find_within_range(*region,
                           back_insert_iterator<vector<kdtreeNode> >(retVec));
    
    //down
    region->_M_low_bounds[0] = p.x - r2;
    region->_M_high_bounds[0] = p.x + r2;
    
    region->_M_low_bounds[1] = p.y - r2;
    region->_M_high_bounds[1] = p.y + r2;
    
    region->_M_low_bounds[2] = p.z - r2;
    region->_M_high_bounds[2] = p.z - r1;
    
    tree.find_within_range(*region,
                           back_insert_iterator<vector<kdtreeNode> >(retVec));
    
    //left
    region->_M_low_bounds[0] = p.x - r2;
    region->_M_high_bounds[0] = p.x - r1;
    
    region->_M_low_bounds[1] = p.y - r2;
    region->_M_high_bounds[1] = p.y + r2;
    
    region->_M_low_bounds[2] = p.z - r1;
    region->_M_high_bounds[2] = p.z + r1;
    
    tree.find_within_range(*region,
                           back_insert_iterator<vector<kdtreeNode> >(retVec));
    
    
    //right
    region->_M_low_bounds[0] = p.x + r1;
    region->_M_high_bounds[0] = p.x + r2;
    
    region->_M_low_bounds[1] = p.y - r2;
    region->_M_high_bounds[1] = p.y + r2;
    
    region->_M_low_bounds[2] = p.z - r1;
    region->_M_high_bounds[2] = p.z + r1;
    
    tree.find_within_range(*region,
                           back_insert_iterator<vector<kdtreeNode> >(retVec));
    

    //front
    region->_M_low_bounds[0] = p.x - r1;
    region->_M_high_bounds[0] = p.x + r1;
    
    region->_M_low_bounds[1] = p.y  - r2;
    region->_M_high_bounds[1] = p.y - r1;
    
    region->_M_low_bounds[2] = p.z - r1;
    region->_M_high_bounds[2] = p.z + r1;
    
    tree.find_within_range(*region,
                           back_insert_iterator<vector<kdtreeNode> >(retVec));
    
    
    //back
    region->_M_low_bounds[0] = p.x - r1;
    region->_M_high_bounds[0] = p.x + r1;
    
    region->_M_low_bounds[1] = p.y  + r1;
    region->_M_high_bounds[1] = p.y + r2;
    
    region->_M_low_bounds[2] = p.z - r1;
    region->_M_high_bounds[2] = p.z + r1;
    
    tree.find_within_range(*region,
                           back_insert_iterator<vector<kdtreeNode> >(retVec));
    
}

void printUsage(string pname){
    printf("Usage: %s <Single Filename>\n",
           pname.c_str()
           );
    printf("-OR-\n");
    printf("%s <prefix> <basename> <numOfFiles> \n",
           pname.c_str()
           );
}




int main(int argv, char * args[]){
    GSnap * psnap;
    
    
    if(argv == 2){
        singlefilename = args[1];
    }else if(argv == 4){
        prefix = args[1];
        basename = args[2];
        numOfFiles = atoi(args[3]);
    }else{
        printUsage(args[0]);
        exit(1);
    }
    
    if(numOfFiles == 0 && singlefilename == ""){
        printUsage(args[0]);
        exit(1);
    }
    
    for(int i = 0; i < NUMBINS; i++){
        z_w[i] = 0;
    }

    if(numOfFiles != 0){
        psnap = new GSnap(prefix, basename, numOfFiles, true, 1 -1);
    }else{
        psnap = new GSnap(singlefilename, true, 1, -1);
    }
    
    allpos = psnap -> getAllPos();
    allvel = psnap -> getAllVel();

    size_t nparts = psnap -> Npart;
    retVec.reserve(nparts);
    
    // make random 3d points
    printf("Building tree ...\n");
    for ( size_t n = 0; n < nparts; ++n)
    {
        kdtreeNode node;
        node.index = n;
        kdtree.insert( node);
    }
    kdtree.optimise();
    
    printf("KDTree Built. Nparts: %ld\n", nparts);
    
    
    for(size_t i = 0; i < nparts; i ++){
        
#ifdef TREE_CODE    
        retVec.clear();
        findPartsInShell(kdtree, allpos[i],
                         radius / SQRT2, radius + shellsize,
                         retVec);

        for(size_t j = 0; j < retVec.size(); j ++){
            //printf("ok1 %d\n", i);
            rvector = allpos[retVec[j].index] - allpos[i];
            //printf("ok2 %d\n", j);
            float r = sqrt(rvector.dot(rvector));
            rvector = rvector / r;
            if(r >= radius && r < radius + shellsize){
                wvector = allvel[retVec[j].index] - allvel[i];
                float w = wvector.dot(rvector);
                
                int ind = (int) floor(w / MAX_W * (NUMBINS / 2) + NUMBINS / 2);
                //printf("%f %d\n", w);
                if(ind >= 0 && ind < NUMBINS){
                    z_w[ind] ++;
                }
            }
        }
#else
        for(size_t j = 0; j < nparts; j ++){
            //printf("ok1 %d\n", i);
            rvector = allpos[j] - allpos[i];
            //printf("ok2 %d\n", j);
            float r = sqrt(rvector.dot(rvector));
            rvector = rvector / r;
            if(r >= radius && r < radius + shellsize){
                wvector = allvel[j] - allvel[i];
                float w = wvector.dot(rvector);
                
                int ind = (int) floor(w / MAX_W * (NUMBINS / 2) + NUMBINS / 2);
                //printf("%f %d\n", w);
                if(ind >= 0 && ind < NUMBINS){
                    z_w[ind] ++;
                }
            }
        }
#endif
        
    }
    
    for(int i = 0; i < NUMBINS; i++){
        float w = (float)(i - NUMBINS / 2) / (float)(NUMBINS / 2) * MAX_W;
        printf("%f %d\n", w, z_w[i]);
    }
    
    delete psnap;
}
