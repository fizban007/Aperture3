#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"

namespace Aperture {

__constant__ SimParamsBase dev_params;
__constant__ Quadmesh dev_mesh;
__constant__ float dev_charges[8];
__constant__ float dev_masses[8];

void init_dev_params(const SimParams& params) {
  CudaSafeCall(cudaMemcpyToSymbol(dev_params, (void*)&params, sizeof(SimParamsBase)));
}

void init_dev_mesh(const Quadmesh& mesh) {
  CudaSafeCall(cudaMemcpyToSymbol(dev_mesh, (void*)&mesh, sizeof(Quadmesh)));
}

void init_dev_charges(const float charges[8]) {
  CudaSafeCall(cudaMemcpyToSymbol(dev_charges, (void*)charges, sizeof(dev_charges)));
}

void init_dev_masses(const float masses[8]) {
  CudaSafeCall(cudaMemcpyToSymbol(dev_masses, (void*)masses, sizeof(dev_masses)));
}

void get_dev_params(SimParams& params) {
  CudaSafeCall(cudaMemcpyFromSymbol((void*)&params, dev_params, sizeof(SimParamsBase)));
}

void get_dev_mesh(Quadmesh& mesh) {
  CudaSafeCall(cudaMemcpyFromSymbol((void*)&mesh, dev_mesh, sizeof(Quadmesh)));
}

void get_dev_charges(float charges[]) {
  CudaSafeCall(cudaMemcpyFromSymbol((void*)charges, dev_charges, sizeof(dev_charges)));
}

void get_dev_masses(float masses[]) {
  CudaSafeCall(cudaMemcpyFromSymbol((void*)masses, dev_masses, sizeof(dev_masses)));
}



}