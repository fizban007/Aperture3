#include "cuda/constant_mem.h"

namespace Aperture {

__constant__ SimParamsBase dev_params;
__constant__ Quadmesh dev_mesh;

}