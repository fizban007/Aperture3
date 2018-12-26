#include "algorithms/field_solver_default.h"
#include "omp.h"
#include "sim_data.h"

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
    size_t k_offset = k * E.data(0).height();
    for (int j = mesh.guard[1]; j < mesh.dims[1] - mesh.guard[1]; j++) {
      size_t offset = (j + k_offset) * E.data(0).pitch();
#pragma omp simd
      for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0];
           i++) {
        size_t global_offset = i * sizeof(Scalar) + offset;
        Scalar E0 = E.data(0)[global_offset];
        Scalar E1 = E.data(1)[global_offset];
        Scalar E2 = E.data(2)[global_offset];
        B.data(0)[global_offset] +=
            -dt *
            ((E2 - E.data(2)[global_offset - E.data(2).pitch()]) *
                 mesh.inv_delta[1] +
             (E1 - E.data(1)[global_offset -
                             E.data(1).pitch() * E.data(1).height()]) *
                 mesh.inv_delta[2]);

        B.data(1)[global_offset] +=
            -dt *
            ((E0 - E.data(0)[global_offset -
                             E.data(0).pitch() * E.data(0).height()]) *
                 mesh.inv_delta[2] +
             (E2 - E.data(2)[global_offset - sizeof(Scalar)]) *
                 mesh.inv_delta[0]);

        B.data(2)[global_offset] +=
            -dt * ((E1 - E.data(1)[global_offset - sizeof(Scalar)]) *
                       mesh.inv_delta[0] +
                   (E0 - E.data(0)[global_offset - E.data(0).pitch()]) *
                       mesh.inv_delta[1]);
      }
    }
  }
}

void
field_solver_default::compute_E_update(vfield_t& E, const vfield_t& B,
                                       const vfield_t& J, double dt) {
  auto& mesh = B.grid().mesh();
  for (int k = mesh.guard[2]; k < mesh.dims[2] - mesh.guard[2]; k++) {
    size_t k_offset = k * B.data(0).height();
    for (int j = mesh.guard[1]; j < mesh.dims[1] - mesh.guard[1]; j++) {
      size_t offset = (j + k_offset) * B.data(0).pitch();
#pragma omp simd
      for (int i = mesh.guard[0]; i < mesh.dims[0] - mesh.guard[0];
           i++) {
        size_t global_offset = i * sizeof(Scalar) + offset;
        Scalar B0 = B.data(0)[global_offset];
        Scalar B1 = B.data(1)[global_offset];
        Scalar B2 = B.data(2)[global_offset];
        E.data(0)[global_offset] +=
            dt *
            ((B2 - B.data(2)[global_offset - B.data(2).pitch()]) *
                 mesh.inv_delta[1] +
             (B1 - B.data(1)[global_offset -
                             B.data(1).pitch() * B.data(1).height()]) *
                 mesh.inv_delta[2] -
             J.data(0)[global_offset]);

        E.data(1)[global_offset] +=
            dt *
            ((B0 - B.data(0)[global_offset -
                             B.data(0).pitch() * B.data(0).height()]) *
                 mesh.inv_delta[2] +
             (B2 - B.data(2)[global_offset - sizeof(Scalar)]) *
                 mesh.inv_delta[0] -
             J.data(1)[global_offset]);

        E.data(2)[global_offset] +=
            dt * ((B1 - B.data(1)[global_offset - sizeof(Scalar)]) *
                      mesh.inv_delta[0] +
                  (B0 - B.data(0)[global_offset - B.data(0).pitch()]) *
                      mesh.inv_delta[1] -
                  J.data(2)[global_offset]);
      }
    }
  }
}

}  // namespace Aperture
