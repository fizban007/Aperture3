#include "algorithms/field_solver_1d.h"
#include "utils/logger.h"

namespace Aperture {

field_solver_1d::field_solver_1d() {}

field_solver_1d::~field_solver_1d() {}

void
field_solver_1d::update_fields(sim_data &data, double dt, double time) {
}

void
field_solver_1d::update_fields(vfield_t &E, const vfield_t &J,
                               const vfield_t &J_bg, double dt,
                               double time) {
  auto &grid = E.grid();
  auto &mesh = grid.mesh();
  // Explicit update
  if (grid.dim() == 1) {
    Logger::print_info("Updating fields");
    for (int i = mesh.guard[0] - 1; i < mesh.dims[0] - mesh.guard[0];
         i++) {
      E(0, i) += dt * (J_bg(0, i) - J(0, i));
    }
  }
}

}  // namespace Aperture
