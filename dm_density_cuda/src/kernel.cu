
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

Tetrahedron * dev_tetras;						//the tetrahedrons in the GPU memory
Tetrahedron * tetras_v;							//the tetrahedrons in the CPU memory
GridManager * gridmanager;						//grid manager
TetraStream * tetrastream;						//tetrahedron stream
REAL * dev_grids;								//the grids in the GPU memory
int * dev_tetra_mem;							//each element specifies the total tetras a block have
int * dev_tetra_select;							//tetra hedron selected in this list
long TETRA_LIST_MEM_LIM = 128*1024*1024;		//128 for the memory lists
int current_tetra_list_ind = 0;					//the current grid block, which is already calculated tetrahedron selection 
int * tetramem;
int * tetramem_list;							//the tetramemory list

//test
//int testmemtttt = 0;

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
		
		/*Point p1;
		p1.x = 139851 % 128 / (REAL) ng * box + dx2; 
		p1.y = 139851 / 128 % 128 / (REAL) ng * box + dx2; 
		p1.z = 139851 / 128 / 128 % 128 / (REAL) ng * box + dx2; 
		Tetrahedron * tetra1 = &dtetra[772]; 
		bool k = tetra1->isInTetra(p1);
		int ggg = tetra_selection[loop_i];
		if(sub_ind == 36 && ggg == 772){

			
			if(k){
				ng ++;
				p1.x ++;
				tetra1->v1.x ++;
			}else{
				ng += ng*1*0;
				k = tetra1->isInTetra(p1);
			}
		}*/
		

		if(tetra->isInTetra(p)){
			//testing
			//dgrids[i + j * sgs + k * sgs * sgs] += 1.0e-11f;
			dgrids[i + j * sgs + k * sgs * sgs] += 1 / tetra->volume;
		}
	}
}

// numsubgrid is the gridsize / subgridsize
// get a actual coordinate of the i, j, k
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

// nsg = gs / subs
// vox_vel = box^3/ng^3;
//check whether the tetrahedron cuboid is in touch with the grid sub-block
__device__ bool isInTouch(int ind, int subgs, int gs, int nsg, float box, float dx2, 
	Tetrahedron * tetra){
	Point v1, v8;
	v1 = getPoint(ind, 0, 0, 0,subgs, gs, nsg, box);
	v8 = getPoint(ind, subgs,subgs,subgs, subgs, gs, nsg, box);

	REAL minx = tetra->minx();
	REAL maxx = tetra->maxx();
	REAL miny = tetra->miny();
	REAL maxy = tetra->maxy();
	REAL minz = tetra->minz();
	REAL maxz = tetra->maxz();

	if (minx > v8.x + dx2 || maxx < v1.x - dx2
		|| miny > v8.y + dx2 || maxy < v1.y - dx2
		|| minz > v8.z + dx2 || maxz < v1.z - dx2){
		return false;
	}
	//test
	/*if (minx > v8.x|| maxx < v1.x
		|| miny > v8.y || maxy < v1.y
		|| minz > v8.z || maxz < v1.z){
		return false;
	}*/

	return true;

}


//compute how many tetrahedrons are in touch with a certain subblock of the density grid
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

//compute the actual list of tetrahedrons thar are in touch with subblock
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
	int total = tetra_mem[ind] - startind;
	for(loop_i = 0; (loop_i < ntetra) && (count < total); loop_i ++){
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
cudaError_t initialCUDA(TetraStream * tetrastream_, GridManager * gridmanager_, int mem_for_tetralist){
	//int grid_size;
	TETRA_LIST_MEM_LIM = mem_for_tetralist;

	tetrastream = tetrastream_;
	gridmanager = gridmanager_;
	//tetras_v = tetrastream_->getTretras();

	num_tetra_ = tetrastream->getBlockSize();
	num_tetra_ = 6 * num_tetra_ * num_tetra_ * num_tetra_;

	sub_grid_size_ = gridmanager->getSubGridSize();
	//grid_size = gridmanager->getGridSize();

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


//compute how many tetrahedrons are in touch with a certain subblock of the density grid
cudaError_t computeTetraMemWithCuda(){
	//copy the memory to CUDA
	cudaError_t cudaStatus;

	tetras_v = tetrastream ->getCurrentBlock();
	num_tetra_ = tetrastream->getBlockNumTetra();

	int blocksize = 512;
	int gridsize = gridmanager->getSubGridNum() / blocksize + 1;

	cudaStatus = cudaMemcpy(dev_tetras, tetras_v, num_tetra_ * sizeof(Tetrahedron), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed -- copying tetrahedrons!\n");
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
		cudaStatus = cudaMemcpy(tetramem_list, dev_tetra_mem, gridmanager->getSubGridNum() * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed -- copying tetra-mem!\n");
			return cudaStatus;
		}
		int j;
		//total_tetra_list_count = gridmanager->getSubGridNum();
		//tetramem_list[gridmanager->getSubGridNum()-1] = tetramem[gridmanager->getSubGridNum()-1];
		tetramem[0] = tetramem_list[0];
		for(j = 1; j < gridmanager->getSubGridNum(); j++){
			if(memoryneed ==0){
				tetramem[j] = tetramem_list[j] + tetramem[j - 1];
				//test
				//printf("%d\n",  tetramem_list[j]);

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
		//at least count 1
		tetramem[current_tetra_list_ind] = tetramem_list[current_tetra_list_ind];
		current_tetra_list_ind ++;
		
		for(j = current_tetra_list_ind; j < gridmanager->getSubGridNum(); j++){
			if(memoryneed ==0){

				//test
				//printf("%d\n",  tetramem_list[j]);

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

	//test
	/*int tttmm[512];
	int tttmm1[512];
	for(int i = 0; i < 512; i++){
		tttmm[i] = tetramem[i];
		tttmm1[i] = tetramem_list[i];
	}*/

	cudaStatus = cudaMemcpy(dev_tetra_mem, tetramem, gridmanager->getSubGridNum() * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed -- copying tetra-mem back!\n");
        return cudaStatus;
    }

	//allocating memory
	//printf("Tetramem: %d\n", tetramem[ gridmanager->getSubGridNum() - 1]);
	int totalmem = memoryneed;

	//test
	//testmemtttt += memoryneed;

	printf("Memory allocating: %d\n", totalmem);
	cudaFree(dev_tetra_select);
	cudaStatus = cudaMalloc((void**)&dev_tetra_select, totalmem * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed -- allocating tetra memory!\n");
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

//density estimation
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
    fprintf(stderr, "cudaMemcpy failed -- copying subgrids!\n");
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
		fprintf(stderr, "cudaMemcpy failed -- copying subgrids!\n");
        return cudaStatus;
    }

	return cudaSuccess;
}

//clean up
void finishCUDA(){
	cudaFree(dev_grids);
	cudaFree(dev_tetras);
	cudaFree(dev_tetra_mem);
	cudaFree(dev_tetra_select);
}
// Helper function for using CUDA to add vectors in parallel.


//assuming the start coordinates is (0,0,0)