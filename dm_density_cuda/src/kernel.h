#ifndef __CUDA_LY_DEN_KERNEL__
#define __CUDA_LY_DEN_KERNEL__

#include "types.h"
#include "kernel.h"
#include "tetrahedron.h"
#include "tetrastream.h"
#include "grids.h"

cudaError_t initialCUDA(TetraStream * tetrastream, GridManager * gridmanager, int mem_for_tetralist);
void finishCUDA();

cudaError_t computeTetraMemWithCuda();

cudaError_t computeTetraSelectionWithCuda(bool & hasmore);

cudaError_t calculateGridWithCuda();
#endif