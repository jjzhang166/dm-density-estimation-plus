/*
 * readgadget.cpp
 *
 *  Created on: Dec 27, 2012
 *      Author: lyang
 */
#include <string>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <algorithm>

using namespace std;

#include "readgadget.h"

struct sort_pred {
    bool operator()(const std::pair<int,int> &left, const std::pair<int,int> &right) {
        return left.first < right.first;
    }
};


GSnap::GSnap(string filename, bool isHighMem, int parttype, int gridsize){
    Npart = 0;
	filename_ = filename;

	uint32_t record0, record1;

	fstream file(filename.c_str(), ios_base::in | ios_base::binary);

	if (!file.good()) {
		printf("File not exist, or corrupted!\n");
		exit(1);
	}

	//read header
	file.read((char *) &record0, sizeof(uint32_t));
	file.read((char *) &header, sizeof(gadget_header));
	file.read((char *) &record1, sizeof(uint32_t));
	if (record0 != record1) {
		printf("Record in file not equal!\n");
		return;
	}

    if(gridsize == -1){
        Npart = header.npart[1];
        grid_size = (int)ceil(pow(Npart, 1.0/3.0));
    }else{
        grid_size = gridsize;
        Npart = grid_size * grid_size * grid_size;
    }
    
    startind=0;
    endind=0;
    for(int i = 0; i < N_TYPE; i++){
        totalparts += header.npart[i];
        if(i < parttype){
            startind += header.npart[i];
        }
    }
    endind = startind + header.npart[parttype];
    
    //printf("%d, %d, %d, %d\n", startind, endind, totalparts, Npart);
    //printf("%d %d %d %d %d %d\n", header.npartTotal[0],header.npartTotal[1], header.npartTotal[2],
    //        header.npartTotal[3],header.npartTotal[4],header.npartTotal[5]);

    isHighMem_ = isHighMem;
    //isHighMem_ = false;
    //read all the data into memory
    if(isHighMem_){
        printf("Loading data into memory...\n");
        
        allind_ = new uint32_t[totalparts];
        allpos_ = new Point[totalparts];
        allvel_ = new Point[totalparts];

        readPos(file, allpos_, 0, totalparts);
        readVel(file, allvel_, 0, totalparts);
        //printf("%d, %d, %f %f %f\n", totalparts, Npart, allpos_[Npart - 1].x, allpos_[Npart - 1].y, allpos_[Npart - 1].z);
        
        //read indexs:
        streamoff spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
			+ sizeof(uint32_t) + totalparts * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t) + totalparts * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t);
	    file.seekg(spos, ios_base::beg);
        file.read((char *) allind_, sizeof(uint32_t) * totalparts);
        //printf("%d\n", allind_[0]);
        
        Point * temppos = allpos_;
        Point * tempvel = allvel_;
        allpos_ = new Point[Npart];
        //allvel_ = new Point[Npart];
        
        
        int indmin = 2147483647;
        for(int i = 0; i < (int)Npart; i ++){
            allpos_[i].x = -1;
            if((int)Npart == (endind - startind)){
                if((int)allind_[i] < indmin){
                    indmin = allind_[i];
                }
            }
        }
        
        //printf("ind min: %d \n", indmin);
        
        for(int i = startind; i < endind; i ++){
            if(allind_[i] >= Npart){
                printf("*************************WARNNING***************************\n");
                printf("particle index is larger than the total number of particles.\nResult is UNPREDICTABLE!!\nPlease double check the indexes, make sure they starts from 0.\n");
                printf("************************************************************\n");
                continue;
            }
            
            if((int)Npart == (endind - startind)){
                allind_[i] -= indmin;
            }
            
            if(allind_[i] >= Npart){
                continue;
            }
            
            allpos_[allind_[i]].x = fmod((float)temppos[i].x, (float)header.BoxSize);
            allpos_[allind_[i]].y = fmod((float)temppos[i].y, (float)header.BoxSize);
            allpos_[allind_[i]].z = fmod((float)temppos[i].z, (float)header.BoxSize);
            //printf("%d %d %d\n", i, allind_[i], totalparts);
        }
        
        //printf("ok\n");
        
        delete temppos;
        allvel_ = new Point[Npart];
        for(int i = startind; i < endind; i ++){
            if(allind_[i] >= Npart){
                continue;
            }
            allvel_[allind_[i]] = tempvel[i];
        }
        
        //printf("ok1\n");
        
        delete allind_;
        delete tempvel;
    }

	file.close();

	//test
	 /*for(int i = 0; i < (int)Npart; i++){
         //printf("%d %f %f %f %f %f %f\n",
         //       ids[i], pos[i*3], pos[i*3 + 1], pos[i*3 + 2],
         //       vel[i*3], vel[i*3+1], vel[i*3+2]);
         if(allpos_[i].x>=0){
             printf("%f %f %f\n", allpos_[i].x, allpos_[i].y, allpos_[i].z);
         }
	 }*/

}


void GSnap::readPosBlock(Point * &posblock, int imin, int jmin, int kmin, int imax, int jmax, int kmax, bool isPeriodical, bool isOrdered){
    int ii = imax - imin + 1;
	int jj = jmax - jmin + 1;
	int kk = kmax - kmin + 1;
    
    if(isHighMem_){
        //copy block memory
        
        for(int j = jmin; j < jmax + 1; j++){
            for(int k = kmin; k < kmax + 1; k++){
                int sindsr = (imin % grid_size) + (j % grid_size) * grid_size + (k % grid_size) * grid_size * grid_size;
                int sinddes = 0 + ((j-jmin)) * ii + ((k-kmin)) * jj * ii;
                int num = ii;
                //printf("%d %d %d\n", sindsr, num, Npart);
                if(sindsr + num < (int)Npart){
                    memcpy((char *)(posblock + sinddes), (char *) (allpos_ + sindsr), num * sizeof(Point));
                }else{
                    memcpy((char *)(posblock + sinddes), (char *) (allpos_ + sindsr), (Npart - sindsr) * sizeof(Point));
                    memcpy((char *)(posblock + sinddes + (Npart - sindsr)), (char *) (allpos_ + sindsr), (num + sindsr -Npart) * sizeof(Point));
                    //printf("Data missed %d %d.\n", Npart - sindsr, num + sindsr -Npart);
                }
                
            }
        }
        return;
    }
    


	int total_cts = ii * jj * kk;

	int * block_count = new int[total_cts];
	//fill_n(total_cts, total_cts, 0);
	int lop_i;
	for(lop_i = 0; lop_i < total_cts; lop_i++){
		block_count[lop_i] = -1;
	}

	//read the positions
	int i, j, k;

    
	fstream file(filename_.c_str(), ios_base::in | ios_base::binary);
	readIndex(file, block_count, imin, jmin, kmin, imax, jmax, kmax, isPeriodical, isOrdered);
	//int kkkk = block_count[511];

	for(i = 0; i < ii; i++){
		for(j = 0; j < jj; j++){
			for(k = 0; k < kk; k++){
                if(block_count[i + j * ii + k * ii * jj] != -1){
                    Point apos = readPos(file, block_count[i + j * ii + k * ii * jj]);
                    posblock[i + j * ii + k * ii * jj].x = fmod((float)apos.x, (float)header.BoxSize);
                    posblock[i + j * ii + k * ii * jj].y = fmod((float)apos.y, (float)header.BoxSize);
                    posblock[i + j * ii + k * ii * jj].z = fmod((float)apos.z, (float)header.BoxSize);
                }else{
                    //mask this point as not usable
                    posblock[i + j * ii + k * ii * jj].x = -1;
                }
			}
		}
	}
    
	file.close();
	delete block_count;
}

void GSnap::readBlock(Point * &posblock, Point * &velocityblock, int imin, int jmin, int kmin, int imax, int jmax, int kmax, 
			bool isPeriodical, bool isOrdered){
    
    if(isHighMem_){
        //copy block memory
        for(int j = jmin; j < jmax; j++){
            for(int k = kmin; k < kmax; k++){
                int startind = (imin % grid_size) + (j % grid_size) * grid_size + (k % grid_size) * grid_size * grid_size;
                int num = imax - imin + 1;
                if(startind + num < (int)Npart){
                    memcpy((char *)(posblock), (char *) (allpos_ + startind), num * sizeof(Point));
                    memcpy((char *)(velocityblock), (char *) (allvel_ + startind), num * sizeof(Point));
                }else{
                    printf("Data missed.\n");
                }
                
            }
        }
        return;
    }
    
	int ii = imax - imin + 1;
	int jj = jmax - jmin + 1;
	int kk = kmax - kmin + 1;

	int total_cts = ii * jj * kk;

	int * block_count = new int[total_cts];
	//fill_n(total_cts, total_cts, 0);
	int lop_i;
	for(lop_i = 0; lop_i < total_cts; lop_i++){
		block_count[lop_i] = -1;
	}

	//read the positions
	int i, j, k;

	fstream file(filename_.c_str(), ios_base::in | ios_base::binary);
	readIndex(file, block_count, imin, jmin, kmin, imax, jmax, kmax, isPeriodical, isOrdered);

	//int kkkk = block_count[511];
	for(i = 0; i < ii; i++){
		for(j = 0; j < jj; j++){
			for(k = 0; k < kk; k++){
                if(block_count[i + j * ii + k * ii * jj] != -1){
                    Point apos = readPos(file, block_count[i + j * ii + k * ii * jj]);
                    posblock[i + j * ii + k * ii * jj].x = fmod((float)apos.x, (float)header.BoxSize);
                    posblock[i + j * ii + k * ii * jj].y = fmod((float)apos.y, (float)header.BoxSize);
                    posblock[i + j * ii + k * ii * jj].z = fmod((float)apos.z, (float)header.BoxSize);

                    apos = readVel(file, block_count[i + j * ii + k * ii * jj]);
                    velocityblock[i + j * ii + k * ii * jj] = apos;
                }else{
                    posblock[i + j * ii + k * ii * jj].x = -1;
                }
			}
		}
	}
	file.close();
	delete block_count;

}

void GSnap::readIndex(std::fstream &file, int *block_count,
		int imin, int jmin, int kmin, int imax, int jmax, int kmax, bool isPeriodical, bool isOrdered){
	//int i = 0;
	int ii = imax - imin + 1;
	int jj = jmax - jmin + 1;
	int kk = kmax - kmin + 1;
	int total_block_num = ii * jj * kk;

	//fseek
    streamoff spos;
    
    //printf("ok1\n");
    if(!isHighMem_){
	    spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
			+ sizeof(uint32_t) + totalparts * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t) + totalparts * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t);
	    file.seekg(spos, ios_base::beg);
    }
    //printf("ok1.5\n");

    {
		int temp_count = 0;
        int i; 
		for(i = 0; i < (int)Npart; i++){
			if(temp_count >= total_block_num){
				break;
			}
			int ind;
            
            //printf("ok3\n");
            if(!isHighMem_){
                file.read((char *) &ind, sizeof(int));              
            }else{
                ind = i;//allind_[i];
            }
            //printf("ok3.5\n");

			int ir = ind % grid_size;
			int jr = (ind / grid_size) % grid_size;
			int kr = (ind / grid_size / grid_size) % grid_size;

			if(!isPeriodical){
				if(jr <= jmax && ir <= imax && kr <= kmax
					&& jr >= jmin && ir >= imin && kr >= kmin){
                        block_count[(ir - imin) + (jr - jmin) * ii + (kr - kmin) * ii * jj] = i;
						temp_count ++;
                }
			}else{
				int inn, jnn, knn;
				for(inn = 0; inn <= 1; inn++){
					for(jnn = 0; jnn <= 1; jnn++){
						for(knn = 0; knn <= 1; knn++){
							int ira, jra, kra;
							ira = ir + inn * grid_size;
							jra = jr + jnn * grid_size;
							kra = kr + knn * grid_size;
							if(jra <= jmax && ira <= imax && kra <= kmax
								&& jra >= jmin && ira >= imin && kra >= kmin){
								block_count[(ira - imin) + (jra - jmin) * ii + (kra - kmin) * ii * jj] = i;
								temp_count ++;
							}
						}
					}
				}

			}
			//find indexes
		}
	}
}

Point GSnap::readPos(std::fstream &file, long ptr){
	Point retp; 

    //printf("ok4\n");
    if(!isHighMem_){
	    streamoff spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
			+ sizeof(uint32_t) + ptr * sizeof(REAL) * 3;
	    file.seekg(spos, ios_base::beg);

	    file.read((char *) &retp.x, sizeof(REAL));
	    file.read((char *) &retp.y, sizeof(REAL));
	    file.read((char *) &retp.z, sizeof(REAL));
    }else{
        retp = allpos_[ptr];
    }
    //printf("ok4.5\n");

	return retp;
}

Point GSnap::readVel(std::fstream &file, long ptr){
	Point retp; 
    if(!isHighMem_){
	    streamoff spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
			+ sizeof(uint32_t) + totalparts * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t) + ptr * sizeof(REAL) * 3;
	    file.seekg(spos, ios_base::beg);

	    file.read((char *) &retp.x, sizeof(REAL));
	    file.read((char *) &retp.y, sizeof(REAL));
	    file.read((char *) &retp.z, sizeof(REAL));
    }else{
        //printf("ok5\n");
        retp = allvel_[ptr];
    }
	return retp;
}

void GSnap::readPos(std::fstream &file, Point * pos, long ptr, long count){
    streamoff spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
			+ sizeof(uint32_t) + ptr * sizeof(REAL) * 3;
	file.seekg(spos, ios_base::beg);
	file.read((char *) pos, sizeof(Point) * count);
    
}
void GSnap::readVel(std::fstream &file, Point * vel, long ptr, long count){
    streamoff spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
			+ sizeof(uint32_t) + totalparts * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t) + ptr * sizeof(REAL) * 3;
    file.seekg(spos, ios_base::beg);
    file.read((char *) vel, sizeof(Point) * count);
}


GSnap::~GSnap() {
	//if (ids != NULL)
	//	delete ids;
    if (isHighMem_){
        //printf("ok\n");
        delete allpos_;
        delete allvel_;
        //delete allind_;
        //printf("ok1\n");
    }
	/*if (pos != NULL)
		delete pos;
	if (vel != NULL)
		delete vel;*/
}

