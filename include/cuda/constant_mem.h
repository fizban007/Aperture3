#ifndef _CONSTANT_MEM_H_
#define _CONSTANT_MEM_H_

#include "sim_params.h"
#include "data/quadmesh.h"


namespace Aperture {

// #ifdef __NVCC__

extern __device__ __constant__ SimParamsBase dev_params;
extern __device__ __constant__ Quadmesh dev_mesh;

// #endif // __NVCC__

}

#endif  // _CONSTANT_MEM_H_
