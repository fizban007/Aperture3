#include "constant_mem_func.h"

namespace Aperture {

// This file is a dummy implementation for initializing the device
// variables, so that we can call these functions even in the CPU
// version without ill effects.

void
init_dev_params(const SimParams& params) {}

void
init_dev_mesh(const Quadmesh& mesh) {}
void
init_dev_charges(const float charges[8]) {}
void
init_dev_masses(const float masses[8]) {}
void
init_dev_bg_fields(vector_field<Scalar>& E, vector_field<Scalar>& B) {}
void
init_dev_rank(int rank) {}

void
get_dev_params(SimParams& params) {}
void
get_dev_mesh(Quadmesh& mesh) {}
void
get_dev_charges(float charges[]) {}
void
get_dev_masses(float masses[]) {}

}  // namespace Aperture
