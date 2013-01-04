
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
__device__ bool isInTouch(int ind, int subgs, int gs, int nsg, float box, float dx2, Tetrahedron * tetra){

	Point p1, p2, p3, p4;
	Point v1, v8;
	//int sg = subgs;

	v1 = getPoint(ind, 0, 0, 0,subgs, gs, nsg, box);
	v8 = getPoint(ind, subgs,subgs,subgs, subgs, gs, nsg, box);


	p1 = tetra->v1;
	p2 = tetra->v2;
	p3 = tetra->v3;
	p4 = tetra->v4;

	double minx = min(min(min(p1.x, p2.x), p3.x), p4.x);
	double maxx = max(max(max(p1.x, p2.x), p3.x), p4.x);
	double miny = min(min(min(p1.y, p2.y), p3.y), p4.y);
	double maxy = max(max(max(p1.y, p2.y), p3.y), p4.y);
	double minz = min(min(min(p1.z, p2.z), p3.z), p4.z);
	double maxz = max(max(max(p1.z, p2.z), p3.z), p4.z);

	if (minx > v8.x - 2*dx2 || maxx < v1.x + 2*dx2
		|| miny > v8.y - 2*dx2 || maxy < v1.y + 2*dx2
		|| minz > v8.z - 2*dx2 || maxz < v1.z + 2*dx2){
		return false;
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
			tetra_mem[ind] += 1;
		}
	}
}


__global__ void computeTetraSelection(Tetrahedron * dtetra, int * tetra_mem, int * tetra_select, 
		int ntetra, int subgridsize, int gridsize, int numsubgrid, float box){
	int loop_i = 0;
	int ind;
	float dx2 = box / gridsize / 2.0;
	ind = blockIdx.x * blockDim.x + threadIdx.x;		//the index of the tetrahedron
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

	//test
	//for(int ffi =0; ffi < num_tetra_; ffi ++){
	//	printf("---%e\n", tetras_v[ffi].volume);
	//}


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
	int * tetramem;
	tetramem = new int[gridmanager->getSubGridNum()];
	cudaStatus = cudaMemcpy(tetramem, dev_tetra_mem, gridmanager->getSubGridNum() * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed -- copying tetra-mem!");
        return cudaStatus;
    }

	int j;
	for(j = 1; j < gridmanager->getSubGridNum(); j++){
		//printf("%d ==> %d\n", j, tetramem[j]);
		tetramem[j] = tetramem[j] + tetramem[j - 1];
		
	}

	cudaStatus = cudaMemcpy(dev_tetra_mem, tetramem, gridmanager->getSubGridNum() * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed -- copying tetra-mem back!");
        return cudaStatus;
    }

	//allocating memory
	//printf("Tetramem: %d\n", tetramem[ gridmanager->getSubGridNum() - 1]);
	int totalmem = tetramem[gridmanager->getSubGridNum() - 1];

	cudaStatus = cudaMalloc((void**)&dev_tetra_select, totalmem * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed -- allocating grids memory!");
        return cudaStatus;
    }



	computeTetraSelection<<<gridsize, blocksize>>>(dev_tetras, dev_tetra_mem, dev_tetra_select, 
		num_tetra_, gridmanager->getSubGridSize(), gridmanager->getGridSize(), 
		gridmanager->getSubGridNum(), 
		gridmanager->getEndPoint().x - gridmanager->getStartPoint().x);

	cudaStatus = cudaThreadSynchronize();
	if( cudaStatus != cudaSuccess){
		fprintf(stderr,"cudaThreadSynchronize error -- sync tetra mem: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	delete tetramem;
	return cudaSuccess;
}

/*
cudaError_t computeTetraSelectionWithCuda(){
	return cudaSuccess;
}
*/

cudaError_t calculateGridWithCuda(){
	cudaError_t cudaStatus;
	//dim3 size(sub_grid_size_, sub_grid_size_, sub_grid_size_);
	dim3 blocksize(8, 8, 8);
	dim3 gridsize(sub_grid_size_/8, sub_grid_size_/8, sub_grid_size_/8);

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
/*
__device__ bool isInTouch(int ind, int subgs, int gs, int nsg, float box, float dx2, Tetrahedron * tetra){

	Point p1, p2, p3, p4;
	Point v1, v2, v3, v4, v5, v6, v7, v8;

	v1 = getPoint(ind, 0, 0, 0, subgs, gs, nsg, box);
	v2 = getPoint(ind, 0, 0,subgs, subgs, gs, nsg, box);
	v3 = getPoint(ind, 0, subgs,0, subgs, gs, nsg, box);
	v4 = getPoint(ind, 0,subgs,subgs, subgs, gs, nsg, box);
	v5 = getPoint(ind, subgs,0, 0, subgs, gs, nsg, box);
	v6 = getPoint(ind, subgs,0,subgs, subgs, gs, nsg, box);
	v7 = getPoint(ind, subgs,subgs,0, subgs, gs, nsg, box);
	v8 = getPoint(ind, subgs,subgs,subgs, subgs, gs, nsg, box);

	p1 = tetra->v1;
	p2 = tetra->v2;
	p3 = tetra->v3;
	p4 = tetra->v4;

	if(isInBox(p1, v1, v8, 2*dx2)
	|| isInBox(p2, v1, v8, 2*dx2)
	|| isInBox(p3, v1, v8, 2*dx2)
	|| isInBox(p4, v1, v8, 2*dx2)){
		return true;
	}

	if(tetra->isInTetra(v1)
	|| tetra->isInTetra(v2)
	|| tetra->isInTetra(v3)
	|| tetra->isInTetra(v4)
	|| tetra->isInTetra(v5)
	|| tetra->isInTetra(v6)
	|| tetra->isInTetra(v7)
	|| tetra->isInTetra(v8))
		return true;
	return false;
}

*/
// dx2 -- the half gridsize





	/*p1.x = minx;
	p1.y = miny;
	p1.z = minz;

	p2.x = minx;
	p2.y = miny;
	p2.z = maxz;

	p3.x = minx;
	p3.y = maxy;
	p3.z = minz;

	p4.x = minx;
	p4.y = maxy;
	p4.z = maxz;

	p5.x = maxx;
	p5.y = miny;
	p5.z = minz;

	p6.x = maxx;
	p6.y = miny;
	p6.z = maxz;

	p7.x = maxx;
	p7.y = maxy;
	p7.z = minz;

	p8.x = maxx;
	p8.y = maxy;
	p8.z = maxz;


	if(isInBox(p1, v1, v8, 2*dx2)
	|| isInBox(p2, v1, v8, 2*dx2)
	|| isInBox(p3, v1, v8, 2*dx2)
	|| isInBox(p4, v1, v8, 2*dx2)
	|| isInBox(p5, v1, v8, 2*dx2)
	|| isInBox(p6, v1, v8, 2*dx2)
	|| isInBox(p7, v1, v8, 2*dx2)
	|| isInBox(p8, v1, v8, 2*dx2)){
		return true;
	}

		//check whether the edge intersects with the cube
	if(minx <= v1.x && maxx >= v8.x){
		if(miny <= v8.y && miny >= v1.y && minz <= v8.z && minz >= v1.z)
			return true;
		if(miny <= v8.y && miny >= v1.y && maxz <= v8.z && maxz >= v1.z)
			return true;
		if(maxy <= v8.y && maxy >= v1.y && minz <= v8.z && minz >= v1.z)
			return true;
		if(maxy <= v8.y && maxy >= v1.y && maxz <= v8.z && maxz >= v1.z)
			return true;
	}

	if(miny <= v1.y && maxy >= v8.y){
		if(minx <= v8.x && minx >= v1.x && minz <= v8.z && minz >= v1.z)
			return true;
		if(minx <= v8.x && minx >= v1.x && maxz <= v8.z && maxz >= v1.z)
			return true;
		if(maxx <= v8.y && maxx >= v1.x && minz <= v8.z && minz >= v1.z)
			return true;
		if(maxx <= v8.x && maxx >= v1.x && maxz <= v8.z && maxz >= v1.z)
			return true;
	}

	if(minz <= v1.z && maxz >= v8.z){
		if(minx <= v8.x && minx >= v1.x && miny <= v8.y && miny >= v1.y)
			return true;
		if(minx <= v8.x && minx >= v1.x && maxy <= v8.y && maxy >= v1.y)
			return true;
		if(maxx <= v8.y && maxx >= v1.x && miny <= v8.y && miny >= v1.y)
			return true;
		if(maxx <= v8.x && maxx >= v1.x && maxy <= v8.y && maxy >= v1.y)
			return true;
	}

	return false;*/
