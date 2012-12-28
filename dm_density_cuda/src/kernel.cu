
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

using namespace std;

#include "types.h"
#include "kernel.h"
#include "tetrahedron.h"
#include "grids.h"

#include "tetrahedron.cu"

int sub_grid_size_;
int num_tetra_;

Tetrahedron * dev_tetras;
REAL * dev_grids;
int * dev_tetra_mem;					//each element specifies the total tetras a block have
int * dev_tetra_select;


__global__ void tetraSplatter(Tetrahedron * dtetra, int ntetra, REAL * dgrids, 
	int gsize, int sub_gsize, REAL, REAL , REAL, REAL);
__global__ void computeTetraMem(Tetrahedron * dtetra){
}


__global__ void tetraSplatter(Tetrahedron * dtetra, int ntetra, REAL * dgrids, 
	int gsize, int sub_gsize, REAL box = 32000, REAL x0 = 0, REAL y0 = 0, REAL z0 = 0){
	int loop_i = 0;
	int i, j, k;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	k = blockIdx.z * blockDim.z + threadIdx.z;
	//ntetra
	for(loop_i = 0; loop_i < ntetra; loop_i ++){
		Tetrahedron * tetra = &dtetra[loop_i];
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


//initialize the CUDA
cudaError_t initialCUDA(int num_tetra, int sub_grid_size){
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }
	sub_grid_size_ = sub_grid_size;
	num_tetra_ = num_tetra;

	// Allocate GPU buffers for tetrahedrons    .
    cudaStatus = cudaMalloc((void**)&dev_tetras, num_tetra * sizeof(Tetrahedron));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed -- allocating tetra memory!");
        return cudaSuccess;
    }

	// Allocate GPU buffers for grids.
    cudaStatus = cudaMalloc((void**)&dev_grids, sub_grid_size_ * sub_grid_size_ * sub_grid_size_ * sizeof(REAL));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed -- allocating grids memory!");
        return cudaSuccess;
    }
	return cudaStatus;
}

cudaError_t calculateGridWithCuda(std::vector<Tetrahedron> * tetras_v, GridManager * gridmanager){
	cudaError_t cudaStatus;
	//dim3 size(sub_grid_size_, sub_grid_size_, sub_grid_size_);
	dim3 blocksize(8, 8, 8);
	dim3 gridsize(sub_grid_size_/8, sub_grid_size_/8, sub_grid_size_/8);

	if(num_tetra_ == 0)
		return cudaErrorUnknown;

	//copy the memory to CUDA
	cudaStatus = cudaMemcpy(dev_tetras, &((*tetras_v)[0]), num_tetra_ * sizeof(Tetrahedron), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed -- copying tetrahedrons!");
        return cudaStatus;
    }

	cudaStatus = cudaMemcpy(dev_grids, gridmanager->getSubGrid(), sub_grid_size_ * sub_grid_size_ * sub_grid_size_ * sizeof(REAL), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed -- copying subgrids!");
        return cudaStatus;
    }
	//<<<1, size>>>
	Point p0 = gridmanager->getPoint(0,0,0);
	tetraSplatter<<<gridsize, blocksize>>>(dev_tetras, num_tetra_, dev_grids, gridmanager->getGridSize(), gridmanager->getSubGridSize(),
		 gridmanager->getEndPoint().x - gridmanager->getStartPoint().x, p0.x, p0.y, p0.z);

	cudaStatus = cudaThreadSynchronize();
	if( cudaStatus != cudaSuccess){
		fprintf(stderr,"cudaThreadSynchronize error: %s\n", cudaGetErrorString(cudaStatus));
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
}
// Helper function for using CUDA to add vectors in parallel.