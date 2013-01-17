#ifndef __CUDA_LY_DEN_KERNEL__
#define __CUDA_LY_DEN_KERNEL__

#include "types.h"
#include "kernel.h"
#include "tetrahedron.h"
#include "tetrastream.h"
#include "gridmanager.h"

cudaError_t initialCUDA(TetraStream * tetrastream, GridManager * gridmanager, int mem_for_tetralist, 
	GridVelocityManager * gridvelocity_ = NULL, bool isVelocity_ = false);

void finishCUDA();

cudaError_t computeTetraMemWithCuda();

cudaError_t computeTetraSelectionWithCuda(bool & hasmore);

cudaError_t calculateGridWithCuda();

#endif