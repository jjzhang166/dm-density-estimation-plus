
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

using namespace std;

#include "types.h"
#include "kernel.h"
#include "tetrahedron.h"
#include "tetrastream.h"
#include "grids.h"

#include "tetrahedron.cu"

using namespace std;

int sub_grid_size_;
int num_tetra_ = 0;

Tetrahedron * dev_tetras;
Tetrahedron * tetras_v;
GridManager * gridmanager;
TetraStream * tetrastream;
REAL * dev_grids;
int * dev_tetra_mem;					//each element specifies the total tetras a block have
int * dev_tetra_select;
long TETRA_LIST_MEM_LIM = 1024*1024*1024;	//1GB for the memory lists
int current_tetra_list_ind = 0;
int total_tetra_list_count = 0;
int * tetramem;
int * tetramem_list;					//the tetramemory list

//__global__ void tetraSplatter(Tetrahedron * dtetra, int ntetra, REAL * dgrids, 
//	int gsize, int sub_gsize, REAL, REAL , REAL, REAL);
//__global__ void computeTetraMem(Tetrahedron * dtetra, int * tetra_mem, int ntetra);
//__global__ void computeTetraSelection(Tetrahedron * dtetra, int * tetra_mem, int * dev_tetra_select);

__global__ void tetraSplatter(Tetrahedron * dtetra, int ntetra, REAL * dgrids,
	int gsize, int sub_gsize, 
	int * tetra_mem, int * tetra_selection, int sub_ind,  int numsubgridsize,
	REAL box = 32000, REAL x0 = 0, REAL y0 = 0, REAL z0 = 0){

	int loop_i = 0;
	int i, j, k;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	k = blockIdx.z * blockDim.z + threadIdx.z;

	if(i >= sub_gsize)
		return;
	if(j >= sub_gsize)
		return;
	if(k >= sub_gsize)
		return;

	int startind = 0;
	if(sub_ind > 0){
		startind = tetra_mem[sub_ind - 1];
	}
	int endind = tetra_mem[sub_ind];

	//ntetra
	//for(loop_i = 0; loop_i < ntetra; loop_i ++){
	for(loop_i = startind; loop_i < endind; loop_i ++){
		Tetrahedron * tetra = &dtetra[tetra_selection[loop_i]];
		if(tetra->maxx() - tetra->minx() > box / 2.0)
			continue;

		if(tetra->maxy() - tetra->miny() > box / 2.0)
			continue;

		if(tetra->maxz() - tetra->minz() > box / 2.0)
			continue;
		
		REAL ng = gsize;

		REAL dx2 = box/ng/2;
		int sgs = sub_gsize;

		//calculate the actual coordinate
		Point p;//getPoint(i, j, k);
		p.x = i / (REAL) ng * box + x0 + dx2; 
		p.y = j / (REAL) ng * box + y0 + dx2;
		p.z = k / (REAL) ng * box + z0 + dx2;


		//testing
		/*
		Point p1;
		p1.x = 12 / (REAL) ng * box + x0 + dx2; 
		p1.y = 5 / (REAL) ng * box + y0 + dx2; 
		p1.z = 11 / (REAL) ng * box + z0 + dx2; 
		Tetrahedron * tetra1 = &dtetra[3165]; 
		bool k = tetra1->isInTetra(p1);
		int ggg = tetra_selection[loop_i];
		if(sub_ind == 25 ){//tetra_selection[loop_i] == 3165){

			
			if(k && (ggg == 3165)){
				ng ++;
				p1.x ++;
				tetra1->v1.x ++;
			}else{
				ng += ng*1*0;
				k = tetra1->isInTetra(p1);
			}
		}
		*/

		if(tetra->isInTetra(p)){
			dgrids[i + j * sgs + k * sgs * sgs] += 1 / tetra->volume;
		}
	}
}

// numsubgrid is the gridsize / subgridsize
__device__ Point getPoint(int ind, int i, int j, int k, int subgridsize, 
		int gridsize, int numsubgrid, float box){
	int ai, aj, ak;

	int i0 = (ind % numsubgrid) * subgridsize;
	int j0 = (ind / numsubgrid % numsubgrid) * subgridsize;
	int k0 = (ind / numsubgrid / numsubgrid % numsubgrid) * subgridsize;

	ai = i + i0;
	aj = j + j0;
	ak = k + k0;

	float fx, fy, fz;

	Point retP;
	fx = (float) ai / (float) gridsize;
	fy = (float) aj / (float) gridsize;
	fz = (float) ak / (float) gridsize;

	retP.x = fx * box;
	retP.y = fy * box;
	retP.z = fz * box;
	return retP;
}


__device__ bool isInBox(Point &p, Point &v1, Point &v2, float dx2){
	return (p.x >= v1.x - dx2 && p.x < v2.x + dx2) && (p.y >= v1.y - dx2 && p.y < v2.y + dx2)
			&& (p.z >= v1.z - dx2 && p.z < v2.z + dx2);
}

// nsg = gs / subs
// vox_vel = box^3/ng^3;
__device__ bool isInTouch(int ind, int subgs, int gs, int nsg, float box, float dx2, 
	Tetrahedron * tetra){

	//Point p1, p2, p3, p4;
	Point v1, v8;
	//int sg = subgs;

	v1 = getPoint(ind, 0, 0, 0,subgs, gs, nsg, box);
	v8 = getPoint(ind, subgs,subgs,subgs, subgs, gs, nsg, box);


	//p1 = tetra->v1;
	//p2 = tetra->v2;
	//p3 = tetra->v3;
	//p4 = tetra->v4;

	double minx = tetra->minx();//min(min(min(p1.x, p2.x), p3.x), p4.x);
	double maxx = tetra->maxx();//max(max(max(p1.x, p2.x), p3.x), p4.x);
	double miny = tetra->miny();//min(min(min(p1.y, p2.y), p3.y), p4.y);
	double maxy = tetra->maxy();//max(max(max(p1.y, p2.y), p3.y), p4.y);
	double minz = tetra->minz();//min(min(min(p1.z, p2.z), p3.z), p4.z);
	double maxz = tetra->maxz();//max(max(max(p1.z, p2.z), p3.z), p4.z);

	if (minx > v8.x + dx2 || maxx < v1.x - dx2
		|| miny > v8.y + dx2 || maxy < v1.y - dx2
		|| minz > v8.z + dx2 || maxz < v1.z - dx2){
		return false;
		//check whether this one is in the vox_vel
		//if((maxx - minx + 1) * ((maxy - miny + 1) * ((maxz - minz + 1))
	}
	return true;

}


__global__ void computeTetraMem(Tetrahedron * dtetra, int * tetra_mem, 
		int ntetra, int subgridsize, int gridsize, int numsubgrid, float box){
	int loop_i = 0;
	int ind;
	float dx2 = box / gridsize / 2.0;
	ind = blockIdx.x * blockDim.x + threadIdx.x;		//the index of the tetrahedron
	if(ind >= numsubgrid){
		return;
	}
	tetra_mem[ind] = 0;
	int subsubgridsize = gridsize / subgridsize;
	for(loop_i = 0; loop_i < ntetra; loop_i ++){
		Tetrahedron * tetra = &(dtetra[loop_i]);
		//check whether the tetra is getting in touch with the current tetra
		if(isInTouch(ind, subgridsize, gridsize, subsubgridsize, box, dx2, tetra)){
			//if(loop_i == 3165)
			tetra_mem[ind] += 1;
		}
	}
}


__global__ void computeTetraSelection(Tetrahedron * dtetra, int * tetra_mem, int * tetra_select, 
		int ntetra, int subgridsize, int gridsize, int numsubgrid, float box,
		int start_ind, int end_ind){
	int loop_i = 0;
	int ind;
	float dx2 = box / gridsize / 2.0;
	ind = blockIdx.x * blockDim.x + threadIdx.x;		//the index of the tetrahedron

	if(ind < start_ind || ind >= end_ind){
		return;
	}

	if(ind >= numsubgrid){
		return;
	}
	int count = 0;
	int startind = 0;
	if(ind > 0){
		startind = tetra_mem[ind - 1];
	}
	int subsubgridsize = gridsize / subgridsize;
	for(loop_i = 0; loop_i < ntetra; loop_i ++){
		Tetrahedron * tetra = &dtetra[loop_i];
		//check whether the tetra is getting in touch with the current tetra
		if(isInTouch(ind, subgridsize, gridsize, subsubgridsize, box, dx2, tetra)){
			//if(loop_i == 3165)
			tetra_select[startind + count] = loop_i;
			count = count + 1;
		}
	}
}

//initialize the CUDA
cudaError_t initialCUDA(TetraStream * tetrastream_, GridManager * gridmanager_){
	int grid_size;

	tetrastream = tetrastream_;
	gridmanager = gridmanager_;
	//tetras_v = tetrastream_->getTretras();

	num_tetra_ = tetrastream->getBlockSize();
	num_tetra_ = 6 * num_tetra_ * num_tetra_ * num_tetra_;

	sub_grid_size_ = gridmanager->getSubGridSize();
	grid_size = gridmanager->getGridSize();

	//printf("%d\n", grid_size);
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

	// Allocate GPU buffers for tetrahedrons    .
    cudaStatus = cudaMalloc((void**)&dev_tetras, num_tetra_ * sizeof(Tetrahedron));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed -- allocating tetra memory!");
        return cudaStatus;
    }

	// Allocate GPU buffers for grids.
    cudaStatus = cudaMalloc((void**)&dev_grids, sub_grid_size_ * sub_grid_size_ * sub_grid_size_ * sizeof(REAL));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed -- allocating grids memory!");
        return cudaStatus;
    }

	// Allocate GPU tetra memory for subgrids.
	int nsub = gridmanager->getSubGridNum();

	cudaStatus = cudaMalloc((void**)&dev_tetra_mem, nsub * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed -- allocating grids memory!");
        return cudaStatus;
    }
	return cudaStatus;
}


cudaError_t computeTetraMemWithCuda(){
	//copy the memory to CUDA
	cudaError_t cudaStatus;

	tetras_v = tetrastream ->getCurrentBlock();
	num_tetra_ = tetrastream->getBlockNumTetra();

	int blocksize = 512;
	int gridsize = gridmanager->getSubGridNum() / blocksize + 1;

	cudaStatus = cudaMemcpy(dev_tetras, tetras_v, num_tetra_ * sizeof(Tetrahedron), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed -- copying tetrahedrons!");
        return cudaStatus;
    }

	//<<<gridsize, blocksize>>>
	computeTetraMem<<<gridsize, blocksize>>>(dev_tetras, dev_tetra_mem, 
		num_tetra_, gridmanager->getSubGridSize(), gridmanager->getGridSize(), 
		gridmanager->getSubGridNum(), 
		gridmanager->getEndPoint().x - gridmanager->getStartPoint().x);

	cudaStatus = cudaThreadSynchronize();
	if( cudaStatus != cudaSuccess){
		fprintf(stderr,"cudaThreadSynchronize error -- sync tetra mem: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	return cudaStatus;
}

//compute the tetraselection
//if has more grid need to be calculate, assign hasmore to be true, otherwise to be false
cudaError_t computeTetraSelectionWithCuda(bool & hasmore){
	int blocksize = 512;
	int gridsize = gridmanager->getSubGridNum() / blocksize + 1;
	cudaError_t cudaStatus;
	int memoryneed = 0;
	int start_index_tetra = current_tetra_list_ind;
	if(current_tetra_list_ind == 0){
		tetramem = new int[gridmanager->getSubGridNum()];
		tetramem_list = new int[gridmanager->getSubGridNum()];
		cudaStatus = cudaMemcpy(tetramem, dev_tetra_mem, gridmanager->getSubGridNum() * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed -- copying tetra-mem!");
			return cudaStatus;
		}
		int j;
		total_tetra_list_count = gridmanager->getSubGridNum();
		tetramem_list[gridmanager->getSubGridNum()-1] = tetramem[gridmanager->getSubGridNum()-1];
		for(j = 1; j < gridmanager->getSubGridNum(); j++){
			tetramem_list[j-1] = tetramem[j-1];
			if(memoryneed ==0){
				tetramem[j] = tetramem[j] + tetramem[j - 1];
				if(tetramem[j] * 4 > TETRA_LIST_MEM_LIM){
					memoryneed = tetramem[j - 1];
					tetramem[j] = memoryneed;
					current_tetra_list_ind = j;
				}
			}else{
				tetramem[j] = memoryneed;
			}
		}

		if(memoryneed == 0){
			memoryneed = tetramem[gridmanager->getSubGridNum()-1];
			current_tetra_list_ind = gridmanager->getSubGridNum();
		}
	}else{
		int j;
		for(j = 0; j < current_tetra_list_ind; j++){
			tetramem[j] = 0;
		}
		for(j = current_tetra_list_ind; j < gridmanager->getSubGridNum(); j++){
			if(memoryneed ==0){
				tetramem[j] = tetramem_list[j] + tetramem[j - 1];
				if(tetramem[j] * 4 > TETRA_LIST_MEM_LIM){
					memoryneed = tetramem[j - 1];
					tetramem[j] = memoryneed;
					current_tetra_list_ind = j;
				}
			}else{
				tetramem[j] = memoryneed;
			}
		}
		if(memoryneed == 0){
			memoryneed = tetramem[gridmanager->getSubGridNum()-1];
			current_tetra_list_ind = gridmanager->getSubGridNum();
		}
	}

	cudaStatus = cudaMemcpy(dev_tetra_mem, tetramem, gridmanager->getSubGridNum() * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed -- copying tetra-mem back!");
        return cudaStatus;
    }

	//allocating memory
	//printf("Tetramem: %d\n", tetramem[ gridmanager->getSubGridNum() - 1]);
	int totalmem = memoryneed;

	cudaStatus = cudaMalloc((void**)&dev_tetra_select, totalmem * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed -- allocating tetra memory!");
        return cudaStatus;
    }

	computeTetraSelection<<<gridsize, blocksize>>>(dev_tetras, dev_tetra_mem, dev_tetra_select, 
		num_tetra_, gridmanager->getSubGridSize(), gridmanager->getGridSize(), 
		gridmanager->getSubGridNum(), 
		gridmanager->getEndPoint().x - gridmanager->getStartPoint().x,
		start_index_tetra, current_tetra_list_ind);

	cudaStatus = cudaThreadSynchronize();
	if( cudaStatus != cudaSuccess){
		fprintf(stderr,"cudaThreadSynchronize error -- sync tetra mem: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	//delete tetramem;
	hasmore = true;
	if(current_tetra_list_ind == gridmanager->getSubGridNum()){
		current_tetra_list_ind = 0;
		hasmore = false;
		delete tetramem;
		delete tetramem_list;
	}
	return cudaSuccess;
}


cudaError_t calculateGridWithCuda(){
	cudaError_t cudaStatus;
	//dim3 size(sub_grid_size_, sub_grid_size_, sub_grid_size_);
	dim3 blocksize(8, 8, 8);
	dim3 gridsize(sub_grid_size_/8, sub_grid_size_/8, sub_grid_size_/8);
	if(sub_grid_size_ %8 != 0){
		gridsize.x ++;
		gridsize.y ++;
		gridsize.z ++;
	}

	if(num_tetra_ == 0)
		return cudaErrorUnknown;

	cudaStatus = cudaMemcpy(dev_grids, gridmanager->getSubGrid(), sub_grid_size_ * sub_grid_size_ * sub_grid_size_ * sizeof(REAL), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed -- copying subgrids!");
        return cudaStatus;
    }
	//<<<1, size>>>
	Point p0 = gridmanager->getPoint(0,0,0);
	//<<<gridsize, blocksize>>>
	tetraSplatter<<<gridsize, blocksize>>>(dev_tetras, num_tetra_, dev_grids, gridmanager->getGridSize(), gridmanager->getSubGridSize(),
		dev_tetra_mem, dev_tetra_select, gridmanager->getCurrentInd(), gridmanager->getSubGridNum(),
		gridmanager->getEndPoint().x - gridmanager->getStartPoint().x, p0.x, p0.y, p0.z);

	cudaStatus = cudaThreadSynchronize();
	if( cudaStatus != cudaSuccess){
		fprintf(stderr,"cudaThreadSynchronize error -- estimate density: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(gridmanager->getSubGrid(), dev_grids, sub_grid_size_ * sub_grid_size_ * sub_grid_size_ * sizeof(REAL), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed -- copying subgrids!");
        return cudaStatus;
    }

	cudaFree(dev_tetra_select);
	return cudaSuccess;
}

void finishCUDA(){
	cudaFree(dev_grids);
	cudaFree(dev_tetras);
	cudaFree(dev_tetra_mem);
	cudaFree(dev_tetra_select);
}
// Helper function for using CUDA to add vectors in parallel.


//assuming the start coordinates is (0,0,0)