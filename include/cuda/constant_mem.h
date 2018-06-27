#ifndef _CONSTANT_MEM_H_
#define _CONSTANT_MEM_H_

#include "sim_params.h"
#include "data/quadmesh.h"


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

void init_dev_params(const SimParams& params);
void init_dev_mesh(const Quadmesh& mesh);
void init_dev_charges(const float charges[8]);
void init_dev_masses(const float masses[8]);

void get_dev_params(SimParams& params);
void get_dev_mesh(Quadmesh& mesh);
void get_dev_charges(float charges[]);
void get_dev_masses(float masses[]);

}

#endif  // _CONSTANT_MEM_H_
