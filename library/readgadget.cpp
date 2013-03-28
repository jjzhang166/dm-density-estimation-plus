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
    
    int startind=0, endind=0;
    for(int i = 0; i < N_TYPE; i++){
        totalparts += header.npart[i];
        if(i < parttype){
            startind += header.npart[i];
        }
    }
    endind = startind + header.npart[parttype];
    
    //printf("%d, %d, %d\n", startind, endind, totalparts);
    //printf("%d %d %d %d %d %d\n", header.npartTotal[0],header.npartTotal[1], header.npartTotal[2],
    //        header.npartTotal[3],header.npartTotal[4],header.npartTotal[5]);

    isHighMem_ = isHighMem;
    //isHighMem_ = false;
    //read all the data into memory
    if(isHighMem_){
        
        allind_ = new uint32_t[totalparts];
        allpos_ = new Point[totalparts];
        allvel_ = new Point[totalparts];

        readPos(file, allpos_, 0, totalparts);
        readVel(file, allvel_, 0, totalparts);
        //printf("%d, %f %f %f\n", totalparts, allvel_[Npart - 1].x, allvel_[Npart - 1].y, allvel_[Npart - 1].z);
        
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
        allvel_ = new Point[Npart];
        for(int i = 0; i < (int)Npart; i ++){
            allpos_[i].x = -1;
        }
        
        for(int i = startind; i < endind; i ++){
            temppos[i].x = fmod((float)temppos[i].x, (float)header.BoxSize);
            temppos[i].y = fmod((float)temppos[i].y, (float)header.BoxSize);
            temppos[i].z = fmod((float)temppos[i].z, (float)header.BoxSize);
            allpos_[allind_[i]] = temppos[i];
            allvel_[allind_[i]] = allvel_[i];
            //printf("%f %f %f\n", allpos_[allind_[i]].x, allpos_[allind_[i]].y, allpos_[allind_[i]].z);
        }
        
        delete allind_;
        delete temppos;
        delete tempvel;
    }

	file.close();

	//test
	/*int i;
	 for(i = 0; i < Npart; i++){
	 printf("%d %f %f %f %f %f %f\n",
	 ids[i], pos[i*3], pos[i*3 + 1], pos[i*3 + 2],
	 vel[i*3], vel[i*3+1], vel[i*3+2]);
	 }

	 exit(0);*/
}


void GSnap::readPosBlock(Point * &posblock, int imin, int jmin, int kmin, int imax, int jmax, int kmax, bool isPeriodical, bool isOrdered){
	int ii = imax - imin + 1;
	int jj = jmax - jmin + 1;
	int kk = kmax - kmin + 1;

	int total_cts = ii * jj * kk;

	int * block_count = new int[total_cts];
	//fill_n(total_cts, total_cts, 0);
	int lop_i;
	for(lop_i = 0; lop_i < total_cts; lop_i++){
		block_count[lop_i] = 0;
	}

	//read the positions
	int i, j, k;

	fstream file(filename_.c_str(), ios_base::in | ios_base::binary);
	readIndex(file, block_count, imin, jmin, kmin, imax, jmax, kmax, isPeriodical, isOrdered);

	//int kkkk = block_count[511];

	for(i = 0; i < ii; i++){
		for(j = 0; j < jj; j++){
			for(k = 0; k < kk; k++){
				Point apos = readPos(file, block_count[i + j * ii + k * ii * jj]);
				posblock[i + j * ii + k * ii * jj] = apos;
			}
		}
	}
	file.close();
	delete block_count;
}

void GSnap::readBlock(Point * &posblock, Point * &velocityblock, int imin, int jmin, int kmin, int imax, int jmax, int kmax, 
			bool isPeriodical, bool isOrdered){
	int ii = imax - imin + 1;
	int jj = jmax - jmin + 1;
	int kk = kmax - kmin + 1;

	int total_cts = ii * jj * kk;

	int * block_count = new int[total_cts];
	//fill_n(total_cts, total_cts, 0);
	int lop_i;
	for(lop_i = 0; lop_i < total_cts; lop_i++){
		block_count[lop_i] = 0;
	}

	//read the positions
	int i, j, k;

	fstream file(filename_.c_str(), ios_base::in | ios_base::binary);
	readIndex(file, block_count, imin, jmin, kmin, imax, jmax, kmax, isPeriodical, isOrdered);

	//int kkkk = block_count[511];
	for(i = 0; i < ii; i++){
		for(j = 0; j < jj; j++){
			for(k = 0; k < kk; k++){
				Point apos = readPos(file, block_count[i + j * ii + k * ii * jj]);
				posblock[i + j * ii + k * ii * jj] = apos;

				apos = readVel(file, block_count[i + j * ii + k * ii * jj]);
				velocityblock[i + j * ii + k * ii * jj] = apos;
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
			+ sizeof(uint32_t) + Npart * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t) + Npart * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t);
	    file.seekg(spos, ios_base::beg);
    }
    //printf("ok1.5\n");

	if(isOrdered){
		int ix, iy, iz;
		for(ix = imin; ix <= imax; ix ++){
			for(iy = jmin; iy <= jmax; iy ++){
				for(iz = kmin; iz <= kmax; iz ++){
					int ix0, iy0, iz0;
					if( ix >= grid_size && isPeriodical){
						ix0 = ix - grid_size;
					}else{
						ix0 = ix;
					}

					if( iy >= grid_size && isPeriodical){
						iy0 = iy - grid_size;
					}else{
						iy0 = iy;
					}

					if( iz >= grid_size && isPeriodical){
						iz0 = iz - grid_size;
					}else{
						iz0= iz;
					}

					int poscount = ix0 + iy0 * grid_size + iz0 * grid_size * grid_size;
                    int ind;

                    //printf("ok2\n");
                    if(!isHighMem_){
					    file.seekg(spos + poscount, ios_base::beg);
                        file.read((char *) &ind, sizeof(int));
                    }else{
                        ind = poscount;//allind_[poscount];
                    }
                    //printf("ok2.5\n");

					block_count[(ix - imin) + (iy - jmin) * ii + (iz - kmin) * ii * jj] = ind;
				}
			}
		}
	}else{
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
                //printf("%d %d\n", i, Npart);
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
			+ sizeof(uint32_t) + Npart * sizeof(REAL) * 3 + sizeof(uint32_t)
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
			+ sizeof(uint32_t) + Npart * sizeof(REAL) * 3 + sizeof(uint32_t)
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

