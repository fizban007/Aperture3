#include "algorithms/current_deposit_Esirkepov.h"
#include "utils/util_functions.h"
#include "data/detail/multi_array_utils.hpp"
#include "sim_environment.h"

namespace Aperture {

namespace Kernels {

__global__
void compute_delta_rho(Scalar** rho, Scalar** delta_rho, )

}

CurrentDepositer_Esirkepov::CurrentDepositer_Esirkepov(const Environment& env)
    : m_env(env) {}

CurrentDepositer_Esirkepov::~CurrentDepositer_Esirkepov() {}

void CurrentDepositer_Esirkepov::deposit(SimData& data, double dt) {
  Logger::print_detail("Depositing current");
  auto& part = data.particles;
  data.J.initialize();

  for (Index_t i = 0; i < data.num_species; i++) {
    data.Rho[i].initialize();
    // data.J_s[i].initialize();
    // data.V[i].initialize();
    // compute_delta_rho(data.J_s[i], data.Rho[i], part[i], dt);
    // normalize_density(data.Rho[i], data.Rho[i]);
  }
  Scalar** rho_ptrs = data.rho_ptrs;
  Scalar** j_ptrs = data.J.array_ptrs();


  // TODO::Handle periodic boundary by copying over the deposited quantities

  // detail::map_multi_array(data.J.data(0), data.J_s[0], data.J.grid().extent(), detail::Op_PlusAssign<Scalar>());
  // communication on the just deposited Rho
  // if (m_comm_rho != nullptr) {
  //   for (Index_t i = 0; i < part.size(); i++) {
  //     m_comm_rho(data.Rho[i]);
  //   }
  // }
  // Now we have delta Q in every cell, add them up along all directions

  scan_current(data.J_s[0]);
  scan_current(data.J_s[1]);
  detail::map_multi_array(data.J.data(0).begin(), data.J_s[0].data().begin(),
                          data.J.grid().extent(), detail::Op_PlusAssign<Scalar>());
  detail::map_multi_array(data.J.data(0).begin(), data.J_s[1].data().begin(),
                          data.J.grid().extent(), detail::Op_PlusAssign<Scalar>());
  // for (unsigned int j = 0; j < part.size(); j++) {
  //   normalize_velocity(data.Rho[j], data.V[j]);
  // }
  // Call communication on just scanned J
  // if (m_comm_J != nullptr) {
  //   m_comm_J(data.J);
  // }

  // TODO::periodic boundary issues
}

void CurrentDepositer_Esirkepov::compute_delta_rho(vfield& J, sfield& Rho,
                                                 const Particles& particles,
                                                 double dt) {
  auto& part = particles.data();
  auto& grid = J.grid();
  if (grid.dim() == 1) {
  }
}

void CurrentDepositer_Esirkepov::compute_delta_rho(sfield& J, sfield& Rho,
                                                 const Particles& particles,
                                                 double dt) {
  auto& part = particles.data();
  auto& grid = J.grid();
  auto charge = particles.charge();
  if (grid.dim() == 1) {
  }
}

void CurrentDepositer_Esirkepov::scan_current(sfield& J) {
  auto& grid = J.grid();
  if (grid.dim() == 1) {
    for (int i = 1; i < grid.mesh().dims[0]; i++) {
      J.data()[i] += J.data()[i-1];
    }
  }
}

void CurrentDepositer_Esirkepov::scan_current(vfield& J) {
  auto& grid = J.grid();
  if (grid.dim() == 1) {
    for (int i = 1; i < grid.mesh().dims[0]; i++) {
      J.data(0)[i] += J.data(0)[i-1];
    }
  }
}

// FIXME: Boundary conditions!
void CurrentDepositer_Esirkepov::normalize_current(const vfield& I, vfield& J) {
  auto& grid = I.grid();
  auto& mesh = I.grid().mesh();
  if (grid.dim() == 1) {
    for (int i = 0; i < mesh.dims[0]; i++) {
      J.data(0)[i] = I.data(0)[i] * mesh.delta[0];
    }
  }
}

void CurrentDepositer_Esirkepov::normalize_density(const sfield& Q,
                                                   sfield& rho, sfield& V) {
}

void CurrentDepositer_Esirkepov::normalize_velocity(const sfield& rho, sfield& V) {
  auto& grid = rho.grid();
  auto& mesh = grid.mesh();
  if (grid.dim() == 1) {
    for (int i = 0; i < mesh.dims[0]; i++) {
      // J.data(0)[i] = I.data(0)[i] * mesh.delta[0];
      if (std::abs(rho(i)) > 1e-5)
        V(i) /= rho(i);
      else
        V(i) = 0.0;
    }
  }
}

}
