#ifndef __CUDA_LY_DEN_KERNEL__
#define __CUDA_LY_DEN_KERNEL__
#include "tetrahedron.h"
#include "grids.h"
cudaError_t initialCUDA(int num_tetra, int grid_size);
void finishCUDA();
cudaError_t calculateGridWithCuda(std::vector<Tetrahedron> * tetras_v, GridManager * gridmanager);
#endif