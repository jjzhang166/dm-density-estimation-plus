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


//#define TREE_CODE

using namespace std;


int num_of_bins = 2000;
double bin_size = 1.0;
double max_w = 1000.0;


#define SQRT3 1.73205080757

//using namespace std;

string singlefilename = "";
string prefix = "";
string basename_ = "";
int numOfFiles = 0;
float radius = 1000.0;
float shellsize = 100.0;
Point wvector;
Point rvector;
uint32_t * z_w;


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
    
    //test
    /*region->_M_low_bounds[0] = p.x - r2;
    region->_M_high_bounds[0] = p.x + r2;
    
    region->_M_low_bounds[1] = p.y - r2;
    region->_M_high_bounds[1] = p.y + r2;
    
    region->_M_low_bounds[2] = p.z - r2;
    region->_M_high_bounds[2] = p.z + r2;
    
    tree.find_within_range(*region,
                           back_insert_iterator<vector<kdtreeNode> >(retVec));
    */
    
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
    fprintf(stderr, "Usage: %s <Single Filename> <radius> <shellsize>\n",
           pname.c_str()
           );
    fprintf(stderr, "-OR-\n");
    fprintf(stderr, "%s <prefix> <basename> <numOfFiles> <radius> <shellsize>\n",
           pname.c_str()
           );
    fprintf(stderr, "Options\n"
                    "-bins <num of bins>, default: 2000\n"
                    "-binsize <size of bin>, default: 1.0 (km/s)\n");
}




int main(int argv, char * args[]){
    GSnap * psnap;
    
    int k = 0;
    if(argv < 4){
        printUsage(args[0]);
        exit(1);
    }else if((argv == 4) || (args[4][0] == '-')){
        singlefilename = args[1];
        stringstream s0, s1;
        s0 << args[2];
        s0 >> radius;
        s1 << args[3];
        s1 >> shellsize;
        
        fprintf(stderr, "Fileame: %s\n", singlefilename.c_str());
        k = 4;
        
    }else if(argv >= 6){
        prefix = args[1];
        basename_ = args[2];
        numOfFiles = atoi(args[3]);

        fprintf(stderr, "Prefix: %s\n", prefix.c_str());
        fprintf(stderr, "BaseName: %s\n", basename_.c_str());
        fprintf(stderr, "Num of Files: %d\n", numOfFiles);
        
        stringstream s0, s1;
        s0 << args[4];
        s0 >> radius;
        s1 << args[5];
        s1 >> shellsize;
        k = 6;
    }else{
        printUsage(args[0]);
        exit(1);
    }
    
   // fprintf(stderr, "%d %d\n", k, argv);
    while(k < argv){
        stringstream ss;
        //fprintf(stderr, "%s\n", args[k]);
        if(strcmp(args[k], "-bins") == 0){
            ss << args[k + 1];
            ss >> num_of_bins; 
        }else if(strcmp(args[k], "-binsize") == 0){
            ss << args[k + 1];
            ss >> bin_size;
        }else{
            printUsage(args[0]);
            exit(1);
        }
        k += 2;
    }

    fprintf(stderr, "radius = %f (kpc / h)\n", radius);
    fprintf(stderr, "shellsize = %f (kpc / h)\n", shellsize);
    fprintf(stderr, "numofbins = %d\n", num_of_bins);
    fprintf(stderr, "bin_size = %f (km/s / h)\n", bin_size);
    
    max_w = num_of_bins / 2 * bin_size;
    
    z_w = new uint32_t[num_of_bins];
    if(numOfFiles == 0 && singlefilename == ""){
        printUsage(args[0]);
        exit(1);
    }
    
    for(int i = 0; i < num_of_bins; i++){
        z_w[i] = 0;
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
    
    //z = 1/a - 1
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
#ifdef TREE_CODE
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
#endif

    
    
    for( size_t i = 0; i < nparts; i ++){
        if(i % m_count == 0){
            fprintf(stderr, ">");
            cout.flush();
        }
#ifdef TREE_CODE    
        retVec.clear();
        findPartsInShell(kdtree, allpos[i],
                         radius / SQRT3 / 1.01, radius + shellsize * 1.01,
                         retVec);
        
        //printf("retsize = %ld, capacit = %ld, nparts = %ld\n", 
        //                retVec.size(), retVec.capacity(), nparts);
#ifdef __OMP__
# pragma omp parallel \
shared ( retVec, allpos, allvel, z_w, radius, shellsize)
        #pragma omp for
#endif
        for(size_t j = 0; j < retVec.size(); j ++){
            //printf("ok1 %d\n", i);
            rvector = allpos[retVec[j].index] - allpos[i];
            //printf("ok2 %d\n", j);
            float r = sqrt(rvector.dot(rvector));
            rvector = rvector / r;
            if(r >= radius && r < radius + shellsize){
                wvector = allvel[retVec[j].index] - allvel[i];
                float w = wvector.dot(rvector);
                
                int ind = (int) floor(w / max_w * (num_of_bins / 2) + num_of_bins / 2);
                //printf("%f %d\n", w);
                if(ind >= 0 && ind < num_of_bins){
#ifdef __OMP__
                    #pragma omp atomic
#endif
                    z_w[ind] ++;
                        
                    
                }
            }
        }
        //printf("\n");
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
                
                int ind = (int) floor(w / max_w * (num_of_bins / 2) + num_of_bins / 2);
                //printf("%f %d\n", w);
                if(ind >= 0 && ind < num_of_bins){
                    z_w[ind] ++;
                }
            }
        }
#endif
        //printf("\n");
    }
    fprintf(stderr, "\n");

    for(int i = 0; i < num_of_bins; i++){
        float w = (float)(i - num_of_bins / 2) / (float)(num_of_bins / 2) * max_w;
        fprintf(stdout, "%f %d\n", w, z_w[i]);
    }
    delete z_w; 
    delete psnap;
}
