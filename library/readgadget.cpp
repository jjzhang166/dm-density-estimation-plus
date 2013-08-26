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
#include <cstring>
#include <utility>
#include <algorithm>
#include <sstream>

using namespace std;

#include "readgadget.h"

struct sort_pred {
    bool operator()(const std::pair<int,int> &left, const std::pair<int,int> &right) {
        return left.first < right.first;
    }
};

void GSnap::init_singlefile(
		string filename,
        bool isHighMem,
        int parttype,
        int gridsize
		){
	isMultifile_ = false;
	isHighMem_ = isHighMem;
    Npart = 0;
	filename_ = filename;
    
	totalparts = 0;
    
	uint32_t record0, record1;
    
	fstream file(filename.c_str(), ios_base::in | ios_base::binary);
    
    //printf("%s\n", filename.c_str());
	if (!file.good()) {
		printf("File not exist, or corrupted!\n");
		exit(1);
	}
    
	//read header
	file.read((char *) &record0, sizeof(uint32_t));
	file.read((char *) &header, sizeof(gadget_header));
	file.read((char *) &record1, sizeof(uint32_t));
    //printf("r0 = %d, r1 = %d headsize = %d\n", (int)record0, (int)record1, sizeof(gadget_header));
	if (record0 != record1) {
		printf("Record in file not equal!\n");
		return;
	}
    
    if(gridsize == -1){
        Npart = header.npart[parttype];
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
			// - indmin
			if(allind_[i] - indmin >= Npart){
				printf("*************************WARNNING***************************\n");
				printf("particle index is larger than the total number of particles.\n"
					   "Result is UNPREDICTABLE!!\nPlease double check the indexes, "
					   "make sure they starts from 0.\n");
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


}

GSnap::GSnap(
      std::string filename,
      int parttype,
      int gridsize
             ){
	init_singlefile(filename, true, parttype, gridsize);
}


GSnap::GSnap(string filename,
             bool isHighMem,
             int parttype,
             int gridsize){
	init_singlefile(filename, isHighMem, parttype, gridsize);
}

//this reads a multi-file into memory
GSnap::GSnap(
             std::string prefix,
             std::string basename,
             int numfiles,
			 bool isHighMem,
             int parttype,
             int gridsize
      ){	
    //printf("%s %s %d\n", prefix.c_str(), basename.c_str(), numfiles);
	isMultifile_ = true;
    numOfFiles_ = numfiles;
    basename_ = basename;
    prefix_ = prefix;
    numOfParts_ = new int[numOfFiles_];
    multStartInd_ = new int[numOfFiles_];
    multEndInd_ = new int[numOfFiles_];
    minInd_ = new int[numOfFiles_];
    maxInd_ = new int[numOfFiles_];
    
    isHighMem_ = isHighMem;
    Npart = 0;
	totalparts = 0;
   
    //printf("ok1.5\n"); 
    for(int i = 0 ; i < numOfFiles_; i ++ ){
        //printf("ok1.5.0\n");
        stringstream ss;
        uint32_t record0, record1;
        string filename = "";// = prefix + basename + ".0";
        //ss << prefix;
        //ss << basename_;
        //ss << ".";
        ss << i;
        //filename_ = filename;
        //ss >> filename;
        
        filename = prefix_ + basename_ + "." + ss.str();
        //printf("ok1.5.0\n");

        //printf("ok1.5.1 filename: %s\n", filename.c_str());
        
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
            exit(1);
        }
        

        
        //printf("ok1.5.2\n");
        numOfParts_[i] = 0;
        multStartInd_[i] = 0;
        //printf("ok1.5.2.1\n");
        for(int j = 0; j < N_TYPE; j++){
            //printf("ok1.5.2.1.0 %d %d %d %d\n", i, j, parttype, numOfFiles_);
            numOfParts_[i] += header.npart[j];
            //printf("ok1.5.2.1.1 %d %d %d %d\n", i, j, parttype, numOfFiles_);
            if(j < parttype){
                multStartInd_[i] += header.npart[j];
            }
            //printf("ok1.5.2.1.2 %d %d %d %d\n", i, j, parttype, numOfFiles_);
        }

        multEndInd_[i] = multStartInd_[i] + header.npart[parttype];
        
        //printf("ok1.5.3\n");
        //find index
        streamoff spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
		+ sizeof(uint32_t) + numOfParts_[i] * sizeof(REAL) * 3 + sizeof(uint32_t)
		+ sizeof(uint32_t) + numOfParts_[i] * sizeof(REAL) * 3 + sizeof(uint32_t)
		+ sizeof(uint32_t) + multStartInd_[i] * sizeof(uint32_t);
        
        
		file.seekg(spos, ios_base::beg);
        
        
        uint32_t * indexs = new uint32_t[header.npart[parttype]];
        
        file.read((char *) indexs, sizeof(uint32_t) * header.npart[parttype]);
        minInd_[i] = header.npartTotal[parttype] + 10000;
        maxInd_[i] = 0;
        for(int j = 0; j < (int)header.npart[parttype]; j++){
            if(minInd_[i] > (int)indexs[j]){
                minInd_[i] = (int)indexs[j];
            }
            if(maxInd_[i] < (int)indexs[j]){
                maxInd_[i] = (int)indexs[j];
            }
        }
        
        delete indexs;

        
        
        file.close();
    
    }

    //printf("ok1.6\n");
    
    if(gridsize == -1){
        Npart = header.npartTotal[parttype];
        grid_size = (int)ceil(pow(Npart, 1.0/3.0));
    }else{
        grid_size = gridsize;
        Npart = grid_size * grid_size * grid_size;
    }
    
    int totalparts = 0;
    for(int i = 0; i < N_TYPE; i++){
        totalparts += header.npartTotal[i];
    }
    
    if(isHighMem_){
        printf("Loading data into memory, %d particles ...\n", Npart);
        
        allind_ = new uint32_t[Npart];
        allpos_ = new Point[Npart];
        allvel_ = new Point[Npart];
        
        for(int i = 0; i < numOfFiles_; i++){
            int single_file_parts = 0;
            ostringstream ss;
            ss << i;
            string filename = prefix + basename + "." + ss.str();
           
            gadget_header single_header;
            
            Point * temppos;
            Point * tempvel;
            uint64_t * tempind;
            int single_startind = 0;
            int single_endind = 0;
            uint32_t record0, record1;
            fstream file(filename.c_str(), ios_base::in | ios_base::binary);
            if(!file.good()){
                printf("File %s corrupted!\n", filename.c_str());
                exit(1);
            }

            file.read((char *) &record0, sizeof(uint32_t));
            file.read((char *) &single_header, sizeof(gadget_header));
            file.read((char *) &record1, sizeof(uint32_t));
           
            if (record0 != record1) {
                printf("Record in file not equal!\n");
                exit(1);
            }
            //file.close();
            
            for(int j = 0; j < N_TYPE; j++){
                single_file_parts += single_header.npart[j];
                if(j < parttype){
                    single_startind += single_header.npart[j];
                }
            }
            
            single_endind = single_startind + single_header.npart[parttype];
            
            
            printf("File %s.%d, particles: %d\n", basename_.c_str(), i, single_file_parts);
            
            temppos = new Point[single_file_parts];
            tempvel = new Point[single_file_parts];
            tempind = new uint64_t[single_file_parts];
            
            readPos(file, temppos, 0, single_file_parts);
            readVel(file, tempvel, 0, single_file_parts);
            
            //read indexs:
            streamoff spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
            + sizeof(uint32_t) + single_file_parts * sizeof(REAL) * 3 + sizeof(uint32_t)
            + sizeof(uint32_t) + single_file_parts * sizeof(REAL) * 3 + sizeof(uint32_t)
            + sizeof(uint32_t);

            file.seekg(spos, ios_base::beg);
            file.read((char *) tempind, sizeof(uint64_t) * single_file_parts);
            
            for(int j = single_startind; j < single_endind; j ++){
                
                //printf("IDs -> j= %d, tempid = %ld, singlparts =%d, nparts = %d\n", 
                //                j, tempind[j], single_file_parts, Npart);
                //printf("%d %d %ld %d\n", j, single_file_parts, tempind[j], totalparts);    
                //printf("Pos -> %f %f %f\n", temppos[j].x, temppos[j].y, temppos[j].z);

                if(tempind[j] >= Npart){
                    continue;
                }
                

                //printf("%d %d %d %d\n", j, single_file_parts, tempind[j], totalparts);
                allpos_[tempind[j]].x = fmod((float)temppos[j].x, (float)header.BoxSize);
                allpos_[tempind[j]].y = fmod((float)temppos[j].y, (float)header.BoxSize);
                allpos_[tempind[j]].z = fmod((float)temppos[j].z, (float)header.BoxSize);
                allvel_[tempind[j]] = tempvel[j];
                //printf("%d %d %d %d\n", j, single_file_parts, tempind[j], totalparts);
            }
            file.close();
            delete temppos;
            delete tempind;
            delete tempvel;
        }
    }
    
    printf("Data loaded!\n");
}


void GSnap::readPosBlock(Point * &posblock,
                         int imin, int jmin, int kmin,
                         int imax, int jmax, int kmax,
                         bool isPeriodical,
                         bool isOrdered)
{
    int ii = imax - imin + 1;
	int jj = jmax - jmin + 1;
	int kk = kmax - kmin + 1;
    
    if(isHighMem_){
        //copy block memory
        //printf("%d\n", ii);
        for(int j = jmin; j < jmax + 1; j++){
            for(int k = kmin; k < kmax + 1; k++){
                int sindsr = (imin % grid_size) + (j % grid_size) * grid_size + (k % grid_size) * grid_size * grid_size;
                int sinddes = 0 + ((j-jmin)) * ii + ((k-kmin)) * jj * ii;
                //printf("%d %d %d\n", sindsr, num, Npart);
                if(imax < grid_size){
                    memcpy((char *)(posblock + sinddes),
                           (char *)(allpos_ + sindsr),
                           ii * sizeof(Point));
                }else{
                    memcpy((char *)(posblock + sinddes),
                           (char *)(allpos_ + sindsr),
                           (grid_size - imin) * sizeof(Point));
                    memcpy((char *)(posblock + sinddes + (grid_size - imin)),
                           (char *)(allpos_ + (j % grid_size) * grid_size + (k % grid_size) * grid_size * grid_size),
                           (imax + 1 - grid_size) * sizeof(Point));
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

void GSnap::readBlock(Point * &posblock, Point * &velocityblock,
                      int imin, int jmin, int kmin,
                      int imax, int jmax, int kmax,
                      bool isPeriodical, bool isOrdered){
    
    int ii = imax - imin + 1;
	int jj = jmax - jmin + 1;
	int kk = kmax - kmin + 1;
    
    if(isHighMem_){
        //copy block memory
        
        for(int j = jmin; j < jmax + 1; j++){
            for(int k = kmin; k < kmax + 1; k++){
                int sindsr = (imin % grid_size) + (j % grid_size) * grid_size + (k % grid_size) * grid_size * grid_size;
                int sinddes = 0 + ((j-jmin)) * ii + ((k-kmin)) * jj * ii;
                //printf("%d %d %d\n", sindsr, num, Npart);
                if(imax < grid_size){
                    memcpy((char *)(posblock + sinddes),
                           (char *)(allpos_ + sindsr),
                           ii * sizeof(Point));
                    memcpy((char *)(velocityblock + sinddes),
                           (char *)(allvel_ + sindsr),
                           ii * sizeof(Point));
                }else{
                    memcpy((char *)(posblock + sinddes),
                           (char *)(allpos_ + sindsr),
                           (grid_size - imin) * sizeof(Point));
                    memcpy((char *)(posblock + sinddes + (grid_size - imin)),
                           (char *)(allpos_ + (j % grid_size) * grid_size + (k % grid_size) * grid_size * grid_size),
                           (imax + 1 - grid_size) * sizeof(Point));
                    
                    memcpy((char *)(velocityblock + sinddes),
                           (char *)(allvel_ + sindsr),
                           (grid_size - imin) * sizeof(Point));
                    memcpy((char *)(velocityblock + sinddes + (grid_size - imin)),
                           (char *)(allvel_ + (j % grid_size) * grid_size + (k % grid_size) * grid_size * grid_size),
                           (imax + 1 - grid_size) * sizeof(Point));
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
                      int imin, int jmin, int kmin,
                      int imax, int jmax, int kmax,
                      bool isPeriodical, bool isOrdered){
	//int i = 0;
	int ii = imax - imin + 1;
	int jj = jmax - jmin + 1;
	int kk = kmax - kmin + 1;
	int total_block_num = ii * jj * kk;

	//fseek
    streamoff spos;
    
    if(!isHighMem_){
	    spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
			+ sizeof(uint32_t) + totalparts * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t) + totalparts * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t);
	    file.seekg(spos, ios_base::beg);
    }

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

void GSnap::readIndex(std::string &file, int *block_count,
                      int imin, int jmin, int kmin,
                      int imax, int jmax, int kmax,
                      bool isPeriodical, bool isOrdered){
    fstream fs(file.c_str(), ios_base::in | ios_base::binary);
    readIndex(fs, block_count,
              imin, jmin, kmin,
              imax, jmax, kmax,
              isPeriodical, isOrdered);
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


Point GSnap::readPos(std::string &file, long ptr){
    fstream fs(file.c_str(), ios_base::in | ios_base::binary);
    return readPos(fs, ptr);
}

Point GSnap::readVel(std::string &file, long ptr){
    fstream fs(file.c_str(), ios_base::in | ios_base::binary);
    return readVel(fs, ptr);
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

void GSnap::readPos(std::string &file,
             Point * pos,
             long ptr,
             long count
             ){
    fstream fs(file.c_str(), ios_base::in | ios_base::binary);
    readPos(fs, pos, ptr, count);
}

void GSnap::readVel(std::fstream &file, Point * vel, long ptr, long count){
    streamoff spos = sizeof(uint32_t) + sizeof(gadget_header) + sizeof(uint32_t)
			+ sizeof(uint32_t) + totalparts * sizeof(REAL) * 3 + sizeof(uint32_t)
			+ sizeof(uint32_t) + ptr * sizeof(REAL) * 3;
    file.seekg(spos, ios_base::beg);
    file.read((char *) vel, sizeof(Point) * count);
}
void GSnap::readVel(std::string &file,
             Point * vel,
             long ptr,
             long count
             ){
    fstream fs(file.c_str(), ios_base::in | ios_base::binary);
    readVel(fs, vel, ptr, count);
}

GSnap::~GSnap() {
    if (isHighMem_){
        delete allpos_;
        delete allvel_;
    }
    if (isMultifile_){
        delete numOfParts_;
        delete multStartInd_;
        delete multEndInd_;
        delete maxInd_;
        delete minInd_;
    }
}

