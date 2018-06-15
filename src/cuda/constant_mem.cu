#include "cuda/constant_mem.h"

namespace Aperture {

__constant__ SimParamsBase dev_params;
__constant__ Quadmesh dev_mesh;
__constant__ float dev_charges[8];
__constant__ float dev_masses[8];

}