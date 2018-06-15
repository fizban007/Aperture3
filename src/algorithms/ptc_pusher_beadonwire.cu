#include "algorithms/ptc_pusher_beadonwire.h"
#include "sim_environment.h"
#include <array>
#include <cmath>
#include <fmt/ostream.h>
#include "utils/logger.h"
#include "utils/util_functions.h"
#include "cuda/cuda_control.h"
#include "cuda/cudaUtility.h"
#include "cuda/constant_mem.h"
#include "algorithms/functions.h"

namespace Aperture {

HD_INLINE double gamma(double beta_phi, double p) {
  double b2 = beta_phi * beta_phi;
  // if (beta_phi < 0) p = -p;

  // if (b2 > 1.0 && p*p/(1.0 + b2) + (1.0 - b2) < 0) {
  //   Logger::print_info("b2 is {}, p is {}, sqrt is {}, {}", b2, p, p*p/(1.0 + b2), (1.0 - b2));
  // }
  // double result = -p * b2 / std::sqrt(1.0 + b2) + std::sqrt(p*p/(1.0 + b2) + (1.0 - b2));
  // result *= 1.0 / (1.0 - b2);

  return std::sqrt(1.0 + p*p + b2);
}


namespace Kernels {

__global__
void lorentz_push(Scalar* p, const Pos_t* x, const uint32_t* cell, const uint32_t* flag,
                  const Scalar* E, double dt, uint32_t num) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < num;
       i += blockDim.x * gridDim.x) {
    if (!check_bit(flag[i], ParticleFlag::ignore_EM)) {
      auto c = cell[i];
      auto rel_x = x[i];
      auto p1 = p[i];
      int sp = get_ptc_type(flag[i]);
      Scalar E1 = E[c] * rel_x + E[c - 1] * (1.0 - rel_x);

      p1 += dev_charges[sp] * E1 * dt / dev_masses[sp];
      p[i] = p1;
    }
  }
}

__global__
void move_ptc(Pos_t* x, uint32_t* cell, const Scalar* p, double dt, uint32_t num) {
  
}

}


ParticlePusher_BeadOnWire::ParticlePusher_BeadOnWire(const Environment& env) :
  m_params(env.params()) {}

ParticlePusher_BeadOnWire::~ParticlePusher_BeadOnWire() {}

void
ParticlePusher_BeadOnWire::push(SimData& data, double dt) {
  Logger::print_info("In particle pusher");
  auto& grid = data.E.grid();
  auto& mesh = grid.mesh();
}

void
ParticlePusher_BeadOnWire::move_ptc(Particles& particles, double x,
                                    const Grid& grid, double dt) {
  auto& ptc = particles.data();
  auto& mesh = grid.mesh();
  if (mesh.dim() == 1) {
  }
}

void
ParticlePusher_BeadOnWire::lorentz_push(Particles& particles, double x,
                                      const VectorField<Scalar>& E,
                                      const VectorField<Scalar>& B, double dt) {
  auto& ptc = particles.data();
  if (E.grid().dim() == 1) {
    Kernels::lorentz_push<<<512, 512>>>(ptc.p1, ptc.x1, ptc.cell, ptc.flag,
                                        E.ptr(0), dt, particles.number());
    CudaCheckError();
  }
}

void
ParticlePusher_BeadOnWire::handle_boundary(SimData &data) {
  auto& mesh = data.E.grid().mesh();
  auto& ptc = data.particles;
  if (ptc.number() > 0) {
    if (m_params.periodic_boundary[0] == false) {
      ptc.clear_guard_cells();
    }
  }
  // auto& photon = data.photons;
  // if (photon.number() > 0) {
  //   if (m_params.periodic_boundary[0] == false) {
  //     photon.clear_guard_cells();
  //   }
  // }

}

void
ParticlePusher_BeadOnWire::extra_force(Particles &particles, Index_t idx, double x, const Grid &grid, double dt) {
  auto& ptc = particles.data();
}

}  // namespace Aperture
