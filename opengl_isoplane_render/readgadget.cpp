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

using namespace std;

#include "readgadget.h"

GSnap::GSnap(string filename) {
	//vel = NULL;
	//pos = NULL;
	ids = NULL;
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

	Npart = header.npart[1];
	grid_size = (int)ceil(pow(Npart, 1.0/3.0));

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
	streamoff spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
			+ sizeof(uint32_t) + Npart * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t) + Npart * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t);
	file.seekg(spos, ios_base::beg);

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
					file.seekg(spos + poscount, ios_base::beg);
					int ind;
					file.read((char *) &ind, sizeof(int));
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
			file.read((char *) &ind, sizeof(int));
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
			/*
			int jma0, jma1, ima0, ima1, kma0, kma1;
			int imi0, imi1, jmi0, jmi1, kmi0, kmi1;
			imi0 = imin;
			jmi0 = jmin;
			kmi0 = kmin;
			imi1 = 0;
			jmi1 = 0;
			kmi1 = 0;

			if(isPeriodical){
				if(imax >= grid_size){
					ima0 = grid_size - 1;
					ima1 = imax % grid_size;
				}else{
					ima1 = -1;
					ima0 = imax;
				}
				
				if(jmax >= grid_size){
					jma0 = grid_size - 1;
					jma1 = jmax % grid_size;
				}else{
					jma1 = -1;
					jma0 = jmax;
				}

				if(kmax >= grid_size){
					kma0 = grid_size - 1;
					kma1 = kmax % grid_size;
				}else{
					kma1 = -1;
					kma0 = kmax;
				}
			}else{
				ima1 = -1;
				jma1 = -1;
				kma1 = -1;
				ima0 = imax;
				jma0 = jmax;
				kma0 = kmax;
			}

			int ai, aj, ak;
			bool isInBlock = true;
			ai = ir;
			aj = jr;
			ak = kr;
			
			if((ai >= imi0) && (ai <= ima0)){
			}else if((ai >= imi1) && (ai <= ima1)){
				 ai = ai + grid_size;
			}else{
				isInBlock = false;
			}

			if(isInBlock){
				if((aj >= jmi0) && (aj <= jma0)){
				}else if((aj >= jmi1) && (aj <= jma1)){
					aj = aj + grid_size;
				}else{
					isInBlock = false;
				}
			}

			if(isInBlock){
				if((ak >= kmi0) && (ak <= kma0)){
				}else if((ak >= kmi1) && (ak <= kma1)){
					ak = ak + grid_size;
				}else{
					isInBlock = false;
				}
			}

			if(isInBlock){
				int b_ind;
				b_ind = (ai - imi0) + (aj - jmi0) * ii + (ak - kmi0) * ii * jj;
				if(b_ind < total_block_num)
					block_count[b_ind] = i;
			}

			if(ii <= grid_size && jj <= grid_size && kk <= grid_size){
				continue;
			}

			//special case
			isInBlock = true;
			ai = ir;
			aj = jr;
			ak = kr;
			
			if((ai >= imi1) && (ai <= ima1)){
				 ai = ai + grid_size;
			}else if((ai >= imi0) && (ai <= ima0)){
			}else{
				isInBlock = false;
			}

			if(isInBlock){
				if((aj >= jmi1) && (aj <= jma1)){
					aj = aj + grid_size;
				}else if((aj >= jmi0) && (aj <= jma0)){
				}else{
					isInBlock = false;
				}
			}

			if(isInBlock){
				if((ak >= kmi1) && (ak <= kma1)){
					ak = ak + grid_size;
				}else if((ak >= kmi0) && (ak <= kma0)){
				}else{
					isInBlock = false;
				}
			}

			if(isInBlock){
				int b_ind;
				b_ind = (ai - imi0) + (aj - jmi0) * ii + (ak - kmi0) * ii * jj;
				if(b_ind < total_block_num)
					block_count[b_ind] = i;
			}*/
		}
	}
}

Point GSnap::readPos(std::fstream &file, long ptr){
	Point retp; 

	streamoff spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
			+ sizeof(uint32_t) + ptr * sizeof(REAL) * 3;
	file.seekg(spos, ios_base::beg);

	file.read((char *) &retp.x, sizeof(REAL));
	file.read((char *) &retp.y, sizeof(REAL));
	file.read((char *) &retp.z, sizeof(REAL));
	return retp;
}

Point GSnap::readVel(std::fstream &file, long ptr){
	Point retp; 

	streamoff spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
			+ sizeof(uint32_t) + Npart * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t) + ptr * sizeof(REAL) * 3;
	file.seekg(spos, ios_base::beg);

	file.read((char *) &retp.x, sizeof(REAL));
	file.read((char *) &retp.y, sizeof(REAL));
	file.read((char *) &retp.z, sizeof(REAL));
	return retp;
}

void GSnap::readPos(std::fstream &file, Point * pos, long ptr, long count){
    streamoff spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
			+ sizeof(uint32_t) + ptr * sizeof(REAL) * 3;
	file.seekg(spos, ios_base::beg);
	file.read((char *) pos, sizeof(REAL) * count);
    
}
void GSnap::readVel(std::fstream &file, Point * vel, long ptr, long count){
    streamoff spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
			+ sizeof(uint32_t) + Npart * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t) + ptr * sizeof(REAL) * 3;
    file.seekg(spos, ios_base::beg);
    file.read((char *) vel, sizeof(REAL) * count);
}


GSnap::~GSnap() {
	if (ids != NULL)
		delete ids;
	/*if (pos != NULL)
		delete pos;
	if (vel != NULL)
		delete vel;*/
}

