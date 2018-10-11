#ifndef _CONSTANT_MEM_H_
#define _CONSTANT_MEM_H_

#include "data/quadmesh.h"
#include "data/field_data.h"
#include "sim_params.h"

namespace Aperture {

// This is the simulation parameters in constant memory
extern __device__ __constant__ SimParamsBase dev_params;
// This is the mesh structure in constant memory
extern __device__ __constant__ Quadmesh dev_mesh;
// This is the charge for each species, up to 8 because we use 3 bits to
// represent species
extern __device__ __constant__ float dev_charges[8];
// This is the mass for each species, up to 8 because we use 3 bits to
// represent species
extern __device__ __constant__ float dev_masses[8];
// This is the structure for background fields
extern __device__ __constant__ FieldData dev_bg_fields;

}  // namespace Aperture

#endif  // _CONSTANT_MEM_H_
