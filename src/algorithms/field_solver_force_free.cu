#include "algorithms/field_solver_force_free.h"
#include "algorithms/finite_diff.h"
#include "data/fields_utils.h"

namespace Aperture {

namespace Kernels {


}

FieldSolver_FFE::FieldSolver_FFE(const Grid& g) :
    m_sf(g), m_tmp(g), m_tmp2(g),
    m_e1(g), m_e2(g), m_e3(g), m_e4(g),
    m_b1(g), m_b2(g), m_b3(g), m_b4(g) {
    // m_j1(g), m_j2(g), m_j3(g), m_j4(g) {
  m_b1.set_field_type(FieldType::B);
  m_b2.set_field_type(FieldType::B);
  m_b3.set_field_type(FieldType::B);
  m_b4.set_field_type(FieldType::B);
}

FieldSolver_FFE::~FieldSolver_FFE() {}

void
FieldSolver_FFE::update_fields(SimData &data, double dt, double time) {
  
}

void
FieldSolver_FFE::compute_J(vfield_t &J, const vfield_t &E, const vfield_t &B) {
  
}

void
FieldSolver_FFE::update_field_substep(vfield_t &E_out, vfield_t &B_out, vfield_t &J_out,
                                      const vfield_t &E_in, const vfield_t &B_in, Scalar dt) {
  // Initialize all tmp fields to zero on the device
  m_tmp.initialize();
  // m_tmp2.initialize();
  m_sf.initialize();

  // Compute the curl of E_in and set it to m_tmp
  m_tmp.set_field_type(FieldType::B);
  curl(m_tmp, E_in);
  // m_tmp2 is now equal to curl E_in
  field_add(B_out, m_tmp, dt);
  ffe_edotb(m_sf, E_in, m_tmp, 1.0f);

  m_tmp.initialize();
  // Compute the curl of B_in and set it to m_tmp
  m_tmp.set_field_type(FieldType::E);
  curl(m_tmp, B_in);
  // m_tmp is now equal to curl B_in
  field_add(E_out, m_tmp, dt);
  ffe_edotb(m_sf, m_tmp, B_in, -1.0f);

  // Now compute FFE current at the cell center
  ffe_j(m_tmp, m_sf, E_in, B_in);
  // TODO: interpolate J back to staggered position
  // TODO: Add J to E_out, conclude the substep update

}


}
