/*
 * readgadget.cpp
 *
 *  Created on: Dec 27, 2012
 *      Author: lyang
 */
#include <string>
#include <fstream>

using namespace std;

#include "readgadget.h"


GSnap::GSnap(string filename){
	vel = NULL;
	pos = NULL;
	ids = NULL;
	Npart = 0;
	uint32_t record0, record1;

	fstream file(filename.c_str(), ios_base::in|ios_base::binary);

	if(!file.good()){
		printf("File not exist, or corrupted!\n");
		exit(1);
	}

	//read header
	file.read((char *) &record0, sizeof(uint32_t));
	file.read((char *) &header, sizeof(gadget_header));
	file.read((char *) &record1, sizeof(uint32_t));
	if(record0 != record1){
		printf("Record in file not equal!\n");
		return;
	}

	Npart = header.npart[1];
	pos = new float[Npart * 3];
	vel = new float[Npart * 3];
	ids = new uint64_t[Npart];

	//read header
	file.read((char *) &record0, sizeof(uint32_t));
	file.read((char *) &header, sizeof(gadget_header));
	file.read((char *) &record1, sizeof(uint32_t));
	if(record0 != record1){
		printf("Record in file not equal!\n");
		return;
	}

	//read pos
	file.read((char *) &record0, sizeof(uint32_t));
	file.read((char *) pos, sizeof(float) * 3 * Npart);
	file.read((char *) &record1, sizeof(uint32_t));
	if(record0 != record1){
		printf("Record in file not equal!\n");
		return;
	}

	//read vel
	file.read((char *) &record0, sizeof(uint32_t));
	file.read((char *) vel, sizeof(float) * 3 * Npart);
	file.read((char *) &record1, sizeof(uint32_t));
	if(record0 != record1){
		printf("Record in file not equal!\n");
		return;
	}

	//read ids
	file.read((char *) &record0, sizeof(uint32_t));
	file.read((char *) ids, sizeof(uint64_t) * Npart);
	file.read((char *) &record1, sizeof(uint32_t));
	if(record0 != record1){
		printf("Record in file not equal!\n");
		return;
	}

	file.close();

}

GSnap::~GSnap(){
	if(ids != NULL) delete ids;
	if(pos != NULL) delete pos;
	if(vel != NULL) delete vel;
}



