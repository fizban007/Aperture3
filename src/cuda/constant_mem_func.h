#ifndef _CONSTANT_MEM_FUNC_H_
#define _CONSTANT_MEM_FUNC_H_

#include "data/fields_dev.h"
#include "data/quadmesh.h"
#include "sim_params.h"

namespace Aperture {

void init_dev_params(const SimParams& params);
void init_dev_mesh(const Quadmesh& mesh);
void init_dev_charges(const float charges[8]);
void init_dev_masses(const float masses[8]);
void init_dev_bg_fields(const cu_vector_field<Scalar>& E,
                        const cu_vector_field<Scalar>& B);

void get_dev_params(SimParams& params);
void get_dev_mesh(Quadmesh& mesh);
void get_dev_charges(float charges[]);
void get_dev_masses(float masses[]);
// void get_dev_bg_fields(cu_vector_field<Scalar>& E, cu_vector_field<Scalar>& B);

}  // namespace Aperture

#endif  // _CONSTANT_MEM_FUNC_H_
