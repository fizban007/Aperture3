#include "algorithms/field_solver_default.h"
#include "sim_data.h"
#include "omp.h"

namespace Aperture {

field_solver_default::field_solver_default(const Grid& g)
    : m_dE(g), m_dB(g) {}

field_solver_default::~field_solver_default() {}

void
field_solver_default::update_fields(vfield_t& E, vfield_t& B,
                                    const vfield_t& J, double dt,
                                    double time) {}

void
field_solver_default::update_fields(sim_data& data, double dt,
                                    double time) {
  update_fields(data.E, data.B, data.J, dt, time);
}

void
field_solver_default::compute_B_update(vfield_t& B, const vfield_t& E,
                                       double dt) {
  auto& mesh = E.grid().mesh();
  for (int k = mesh.guard[2]; k < mesh.dims[2] - mesh.guard[2]; k++) {
    for (int j = mesh.guard[1]; j < mesh.dims[1] - mesh.guard[1]; j++) {
#pragma omp simd
      for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0]; i++) {
        
      }
    }
  }
}

void
field_solver_default::compute_E_update(vfield_t& E, const vfield_t& B,
                                       const vfield_t& J, double dt) {}

}  // namespace Aperture
