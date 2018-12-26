#include "data/detail/multi_array_utils.hpp"
#include "field_solver.h"

using namespace Aperture;

FieldSolver_FiniteDiff::FieldSolver_FiniteDiff(const Grid &g,
                                               double alpha)
    : m_dE(g),
      m_dB(g),
      m_E_tmp(g),
      m_B_old(g),
      m_vfield_tmp(g),
      m_sfield_tmp(g),
      m_alpha(alpha),
      m_beta(1.0 - alpha) {
  m_dE.set_field_type(FieldType::E);
  m_dB.set_field_type(FieldType::B);
  m_E_tmp.set_field_type(FieldType::E);
  m_B_old.set_field_type(FieldType::B);
}

FieldSolver_FiniteDiff::~FieldSolver_FiniteDiff() {}

void
FieldSolver_FiniteDiff::compute_laplacian(
    const vfield &input, vfield &output, const Index &start,
    const Extent &ext, const bool is_boundary[], int order) {
  // Invoke the series of differential operators
  FiniteDiff::compute_curl(input, m_vfield_tmp, is_boundary, start, ext,
                           order);
  if (m_comm_callback_vfield != nullptr)
    m_comm_callback_vfield(m_vfield_tmp);
  m_vfield_tmp.set_stagger(input.stagger_dual());

  FiniteDiff::compute_divergence(input, m_sfield_tmp, is_boundary,
                                 start, ext, order);
  if (m_comm_callback_sfield != nullptr)
    m_comm_callback_sfield(m_sfield_tmp);

  FiniteDiff::compute_curl(m_vfield_tmp, output, is_boundary, start,
                           ext, order);
  FiniteDiff::compute_gradient(m_sfield_tmp, m_vfield_tmp, is_boundary,
                               start, ext, order);

  // Put both components together
  for (int comp = 0; comp < 3; comp++) {
    subtract(output.data(comp).begin(), m_vfield_tmp.data(comp).begin(),
             output.grid().mesh().extent());
    multiply(output.data(comp).begin(), output.grid().mesh().extent(),
             -1.0);
  }

  // FIXME: Is this really needed?
  if (m_comm_callback_vfield != nullptr) m_comm_callback_vfield(output);
}

// Implicit field updater
void
FieldSolver_FiniteDiff::compute_B_update(vfield &B, const vfield &E,
                                         const vfield &J, double dt) {
  const int num_expansion = 5;
  // Assuming B, E, and J are already communicated
  m_B_old.copy_from(B);
  m_dB.assign(0.0);
  if (std::abs(m_beta) > EPS) {
    compute_laplacian(B, m_dB, m_bc->bound_start(), m_bc->bound_ext(),
                      m_bc->is_boundary());
    B.addBy(m_dB.multiplyBy(m_alpha * m_beta * dt * dt));
  }

  // FIXME: Boundary adjust issue
  FiniteDiff::compute_curl(E, m_dB, m_bc->is_boundary(),
                           m_bc->bound_start(), m_bc->bound_ext());
  B.addBy(m_dB.multiplyBy(-1.0 * dt));

  FiniteDiff::compute_curl(J, m_dB, m_bc->is_boundary(),
                           m_bc->bound_start(), m_bc->bound_ext());
  B.addBy(m_dB.multiplyBy(m_alpha * dt * dt));

  if (m_comm_callback_vfield != nullptr) m_comm_callback_vfield(B);

  m_dB.copy_from(B);
  double factor = dt * dt * m_alpha * m_alpha;
  for (int j = 0; j < num_expansion; j++) {
    m_dB.multiplyBy(factor);
    compute_laplacian(m_dB, m_dB, m_bc->bound_start(),
                      m_bc->bound_ext(), m_bc->is_boundary());
    B.addBy(m_dB);

    if (m_comm_callback_vfield != nullptr) m_comm_callback_vfield(m_dB);
  }

  if (m_comm_callback_vfield != nullptr) m_comm_callback_vfield(B);
}

void
FieldSolver_FiniteDiff::compute_E_update(vfield &E, const vfield &B,
                                         const vfield &J, double dt) {
  // Assume communication is already done
  FiniteDiff::compute_curl(B, m_E_tmp, m_bc->is_boundary(),
                           m_bc->bound_start(), m_bc->bound_ext());
  m_dE.addBy(m_E_tmp.multiplyBy(m_alpha));

  FiniteDiff::compute_curl(m_B_old, m_E_tmp, m_bc->is_boundary(),
                           m_bc->bound_start(), m_bc->bound_ext());
  m_dE.addBy(m_E_tmp.multiplyBy(m_beta));

  m_dE.subtractBy(J);
  m_dE.multiplyBy(dt);
  E.addBy(m_dE);

  if (m_comm_callback_vfield != nullptr) m_comm_callback_vfield(E);
}

void
FieldSolver_FiniteDiff::update_fields(vfield &E, vfield &B,
                                      const vfield &J, double dt,
                                      double time) {
  compute_B_update(B, E, J, dt);
  compute_E_update(B, E, J, dt);
}

void
FieldSolver_FiniteDiff::update_fields(Aperture::SimData &data,
                                      double dt, double time) {}

void
FieldSolver_FiniteDiff::compute_flux(const vfield &f, sfield &flux) {}
