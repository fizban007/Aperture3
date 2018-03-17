#include "algorithms/field_solver_integral.h"

using namespace Aperture;

FieldSolver_Integral::FieldSolver_Integral(const Grid &g, const Grid &g_dual)
    : m_dE(g), m_dB(g_dual), m_background_j(g) {
  m_background_j.initialize();
}

FieldSolver_Integral::~FieldSolver_Integral() {}

void
FieldSolver_Integral::update_fields(vfield_t &E, vfield_t &B, const vfield_t &J,
                                    double dt, double time) {
  // Logger::print_info("Updating fields");
  auto &grid = E.grid();
  auto &mesh = grid.mesh();
  // Explicit update
  if (grid.dim() == 1) {
    // for (int i = 0; i < mesh.dims[0]; i++) {
    for (int i = mesh.guard[0] - 1; i < mesh.dims[0] - mesh.guard[0]; i++) {
      // TODO: Add a background J?
      E(0, i) += dt * (m_background_j(0, i) - J(0, i));
    }
    // for (int i = 0; i < mesh.gu)
  }

  if (m_comm_callback_vfield != nullptr) {
    m_comm_callback_vfield(E);
  }
}

void
FieldSolver_Integral::update_fields(Aperture::SimData &data, double dt,
                                    double time) {
  update_fields(data.E, data.B, data.J, dt, time);
  data.B.addBy(data.E);
}

void
FieldSolver_Integral::set_background_j(const vfield_t &j) {
  m_background_j = j;
}
