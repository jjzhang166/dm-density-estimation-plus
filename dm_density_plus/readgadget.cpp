/*
 * readgadget.cpp
 *
 *  Created on: Dec 27, 2012
 *      Author: lyang
 */
#include <string>
#include <fstream>
#include <cmath>

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
	grid_size = ceil(pow(Npart, 1.0/3.0));
	/*pos = new float[Npart * 3];
	vel = new float[Npart * 3];
	ids = new uint32_t[Npart];

	//read pos
	file.read((char *) &record0, sizeof(uint32_t));
	file.read((char *) pos, sizeof(float) * 3 * Npart);
	file.read((char *) &record1, sizeof(uint32_t));
	if (record0 != record1) {
		printf("Record in file not equal--pos!\n");
		return;
	}

	//read vel
	file.read((char *) &record0, sizeof(uint32_t));
	file.read((char *) vel, sizeof(float) * 3 * Npart);
	file.read((char *) &record1, sizeof(uint32_t));
	if (record0 != record1) {
		printf("Record in file not equal--vel!\n");
		return;
	}

	//read ids
	file.read((char *) &record0, sizeof(uint32_t));
	file.read((char *) ids, sizeof(uint32_t) * Npart);
	file.read((char *) &record1, sizeof(uint32_t));
	if (record0 != record1) {
		printf("Record in file not equal--ids!\n");
		return;
	}*/

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


void GSnap::readPosBlock(Point * posblock, int imin, int jmin, int kmin, int imax, int jmax, int kmax){
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
	readIndex(file, block_count, imin, jmin, kmin, imax, jmax, kmax);

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

void GSnap::readIndex(std::fstream &file, int *block_count,
		int imin, int jmin, int kmin, int imax, int jmax, int kmax){
	int i;
	int ii = imax - imin + 1;
	int jj = jmax - jmin + 1;
	//int kk = kmax - kmin + 1;

	//fseek
	streamoff spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
			+ sizeof(uint32_t) + Npart * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t) + Npart * sizeof(REAL) * 3 + sizeof(uint32_t);
	file.seekg(spos, ios_base::beg);

	for(i = 0; i < (int)Npart; i++){
		int ind;
		file.read((char *) &ind, sizeof(int));
		int ir = ind % grid_size;
		int jr = (ind / grid_size) % grid_size;
		int kr = (ind / grid_size / grid_size) % grid_size;
		//find indexes
		if(jr <= jmax && ir <= imax && kr <= kmax
		&& jr >= jmin && ir >= imin && kr >= kmin){
			block_count[(ir - imin) + (jr - jmin) * ii + (kr - kmin) * ii * jj] = i;
		}
	}
}

Point GSnap::readPos(std::fstream &file, int count){
	Point retp;

	streamoff spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
			+ sizeof(uint32_t) + count * sizeof(REAL) * 3;
	file.seekg(spos, ios_base::beg);

	file.read((char *) &retp.x, sizeof(REAL));
	file.read((char *) &retp.y, sizeof(REAL));
	file.read((char *) &retp.z, sizeof(REAL));
	return retp;
}

GSnap::~GSnap() {
	if (ids != NULL)
		delete ids;
	/*if (pos != NULL)
		delete pos;
	if (vel != NULL)
		delete vel;*/
}

