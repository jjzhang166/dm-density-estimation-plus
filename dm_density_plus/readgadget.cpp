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
	ids = new uint32_t[Npart];


	//read pos
	file.read((char *) &record0, sizeof(uint32_t));
	file.read((char *) pos, sizeof(float) * 3 * Npart);
	file.read((char *) &record1, sizeof(uint32_t));
	if(record0 != record1){
		printf("Record in file not equal--pos!\n");
		return;
	}

	//read vel
	file.read((char *) &record0, sizeof(uint32_t));
	file.read((char *) vel, sizeof(float) * 3 * Npart);
	file.read((char *) &record1, sizeof(uint32_t));
	if(record0 != record1){
		printf("Record in file not equal--vel!\n");
		return;
	}

	//read ids
	file.read((char *) &record0, sizeof(uint32_t));
	file.read((char *) ids, sizeof(uint32_t) * Npart);
	file.read((char *) &record1, sizeof(uint32_t));
	if(record0 != record1){
		printf("Record in file not equal--ids!\n");
		return;
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

GSnap::~GSnap(){
	if(ids != NULL) delete ids;
	if(pos != NULL) delete pos;
	if(vel != NULL) delete vel;
}



