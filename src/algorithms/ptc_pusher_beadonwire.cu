#include "algorithms/functions.h"
#include "algorithms/ptc_pusher_beadonwire.h"
#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
#include "cuda/cuda_control.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/util_functions.h"
#include <array>
#include <cmath>

namespace Aperture {

HD_INLINE double
gamma(double beta_phi, double p) {
  double b2 = beta_phi * beta_phi;
  // if (beta_phi < 0) p = -p;

  // if (b2 > 1.0 && p*p/(1.0 + b2) + (1.0 - b2) < 0) {
  //   Logger::print_info("b2 is {}, p is {}, sqrt is {}, {}", b2, p,
  //   p*p/(1.0 + b2), (1.0 - b2));
  // }
  // double result = -p * b2 / std::sqrt(1.0 + b2) + std::sqrt(p*p/(1.0
  // + b2) + (1.0 - b2)); result *= 1.0 / (1.0 - b2);

  return std::sqrt(1.0 + p * p + b2);
}

namespace Kernels {

// TODO: consider fusing these kernels?

__global__ void
lorentz_push(particle_data ptc, const Scalar* E,
             Grid::const_mesh_ptrs mp, double dt, uint32_t num) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    if (!check_bit(ptc.flag[i], ParticleFlag::ignore_EM)) {
      auto c = ptc.cell[i];
      // Skip empty particles
      if (c == MAX_CELL) continue;
      auto rel_x = ptc.x1[i];
      auto p1 = ptc.p1[i];
      int sp = get_ptc_type(ptc.flag[i]);
      Scalar E1 = E[c] * rel_x + E[c - 1] * (1.0 - rel_x);

      p1 += dev_charges[sp] * E1 * dt / dev_masses[sp];
      ptc.p1[i] = p1;
    }
  }
}

__global__ void
move_ptc(particle_data ptc, Grid::const_mesh_ptrs mp, double dt,
         uint32_t num) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    auto c = ptc.cell[i];
    // Skip empty particles
    if (c == MAX_CELL) continue;
    auto rel_x = ptc.x1[i];
    auto p = ptc.p1[i];
    // Scalar gamma = sqrt(1.0 + p*p);
    Scalar a2 = mp.a2[c] * rel_x + mp.a2[c - 1] * (1.0 - rel_x);
    Scalar D1 = mp.D1[c] * rel_x + mp.D1[c - 1] * (1.0 - rel_x);
    Scalar D2 = mp.D2[c] * rel_x + mp.D2[c - 1] * (1.0 - rel_x);
    Scalar D3 = mp.D3[c] * rel_x + mp.D3[c - 1] * (1.0 - rel_x);
    Scalar u0 =
        std::sqrt((1.0 + p * p / D2) / (a2 - D3 + D1 * D1 / D2));

    // Scalar dx = p * dt / (gamma * dev_mesh.delta[0]);
    Scalar dx = dt * (p / u0 - D1) / (D2 * dev_mesh.delta[0]);
    Scalar new_x1 = ptc.x1[i] + dx;
    int delta_c = floor(new_x1);
    c += delta_c;

    // ptc.dx1[i] = dx;
    ptc.cell[i] = c;
    ptc.x1[i] = new_x1 - (Pos_t)delta_c;
  }
}

__global__ void
move_photon(photon_data photon, Grid::const_mesh_ptrs mp, double dt,
            uint32_t num) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    auto c = photon.cell[i];
    // Skip empty particles
    if (c == MAX_CELL) continue;
    auto rel_x = photon.x1[i];
    auto p = photon.p1[i];
    // Scalar gamma = sqrt(1.0 + p*p);
    Scalar a2 = mp.a2[c] * rel_x + mp.a2[c - 1] * (1.0 - rel_x);
    Scalar D1 = mp.D1[c] * rel_x + mp.D1[c - 1] * (1.0 - rel_x);
    Scalar D2 = mp.D2[c] * rel_x + mp.D2[c - 1] * (1.0 - rel_x);
    Scalar D3 = mp.D3[c] * rel_x + mp.D3[c - 1] * (1.0 - rel_x);
    Scalar u0 = std::sqrt((p * p / D2) / (a2 - D3 + D1 * D1 / D2));

    // Scalar dx = p * dt / (gamma * dev_mesh.delta[0]);
    // TODO: fix pitch angle thing!
    Scalar dx = dt * (p / u0 - D1) / (D2 * dev_mesh.delta[0]);
    Scalar new_x1 = photon.x1[i] + dx;
    int delta_c = floor(new_x1);
    c += delta_c;

    photon.cell[i] = c;
    photon.x1[i] = new_x1 - (Pos_t)delta_c;
    photon.path_left[i] -= std::abs(dx);
  }
}

__global__ void
push_and_move(particle_data ptc, const Scalar* E,
              Grid::const_mesh_ptrs mp, double dt, uint32_t num) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    auto c = ptc.cell[i];
    // Skip empty particles
    if (c == MAX_CELL) continue;
    auto rel_x = ptc.x1[i];
    auto p = ptc.p1[i];

    // First update p
    int sp = get_ptc_type(ptc.flag[i]);
    Scalar E1 =
        mp.alpha_grr[c] * E[c] * rel_x / mp.A[c] +
        mp.alpha_grr[c - 1] * E[c - 1] * (1.0 - rel_x) / mp.A[c - 1];

    Scalar a2 = mp.a2[c] * rel_x + mp.a2[c - 1] * (1.0 - rel_x);
    Scalar D1 = mp.D1[c] * rel_x + mp.D1[c - 1] * (1.0 - rel_x);
    Scalar D2 = mp.D2[c] * rel_x + mp.D2[c - 1] * (1.0 - rel_x);
    Scalar D3 = mp.D3[c] * rel_x + mp.D3[c - 1] * (1.0 - rel_x);
    Scalar u0 =
        std::sqrt((1.0 + p * p / D2) / (a2 - D3 + D1 * D1 / D2));
    Scalar v = (p / u0 - D1) / D2;

    Scalar drda2 = (mp.a2[c] - mp.a2[c - 1]) / dev_mesh.delta[0];
    Scalar drdD1 = (mp.D1[c] - mp.D1[c - 1]) / dev_mesh.delta[0];
    Scalar drdD2 = (mp.D2[c] - mp.D2[c - 1]) / dev_mesh.delta[0];
    Scalar drdD3 = (mp.D3[c] - mp.D3[c - 1]) / dev_mesh.delta[0];

    p += dev_charges[sp] * E1 * dt / dev_masses[sp];
    p -= 0.5 * u0 *
         (drda2 - (drdD3 + 2.0 * v * drdD1 + v * v * drdD2)) * dt;
    ptc.p1[i] = p;

    // Then compute dx and update x
    u0 = std::sqrt((1.0 + p * p / D2) / (a2 - D3 + D1 * D1 / D2));
    v = (p / u0 - D1) / D2;
    Scalar dx = dt * v / dev_mesh.delta[0];
    Scalar new_x1 = rel_x + dx;
    int delta_c = floor(new_x1);
    c += delta_c;

    // ptc.dx1[i] = dx;
    ptc.cell[i] = c;
    ptc.x1[i] = new_x1 - (Pos_t)delta_c;
  }
}

}  // namespace Kernels

ParticlePusher_BeadOnWire::ParticlePusher_BeadOnWire(
    const Environment& env)
    : m_params(env.params()) {}

ParticlePusher_BeadOnWire::~ParticlePusher_BeadOnWire() {}

void
ParticlePusher_BeadOnWire::push(SimData& data, double dt) {
  Logger::print_info("Pushing Particles");
  auto& grid = data.E.grid();
  auto& mesh = grid.mesh();

  lorentz_push(data.particles, data.E, data.B, dt);
  move_photons(data.photons, grid, dt);
  // move_ptc(data.particles, grid, dt);
}

void
ParticlePusher_BeadOnWire::move_ptc(Particles& particles,
                                    const Grid& grid, double dt) {
  auto& ptc = particles.data();
  auto& mesh = grid.mesh();
  if (mesh.dim() == 1) {
    Kernels::move_ptc<<<512, 512>>>(ptc, grid.get_mesh_ptrs(), dt,
                                    particles.number());
    CudaCheckError();
  }
}

void
ParticlePusher_BeadOnWire::move_photons(Photons& photons,
                                        const Grid& grid, double dt) {
  auto& ph = photons.data();
  auto& mesh = grid.mesh();
  if (mesh.dim() == 1) {
    Kernels::move_photon<<<512, 512>>>(ph, grid.get_mesh_ptrs(), dt,
                                       photons.number());
    CudaCheckError();
  }
}

void
ParticlePusher_BeadOnWire::lorentz_push(Particles& particles,
                                        const VectorField<Scalar>& E,
                                        const VectorField<Scalar>& B,
                                        double dt) {
  auto& ptc = particles.data();
  auto& grid = E.grid();
  if (E.grid().dim() == 1) {
    // Kernels::lorentz_push<<<512, 512>>>(ptc, E.ptr(0),
    // grid.get_mesh_ptrs(), dt, particles.number());
    // Kernels::push_and_move<<<512, 512>>>(ptc, E.ptr(0),
    // grid.get_mesh_ptrs(), dt, particles.number()); CudaCheckError();
  }
}

void
ParticlePusher_BeadOnWire::handle_boundary(SimData& data) {
  auto& mesh = data.E.grid().mesh();
  auto& ptc = data.particles;
  if (ptc.number() > 0) {
    if (m_params.periodic_boundary[0] == false) {
      ptc.clear_guard_cells();
    }
  }
  auto& photon = data.photons;
  if (photon.number() > 0) {
    if (m_params.periodic_boundary[0] == false) {
      photon.clear_guard_cells();
    }
  }
}

void
ParticlePusher_BeadOnWire::extra_force(Particles& particles,
                                       Index_t idx, double x,
                                       const Grid& grid, double dt) {
  auto& ptc = particles.data();
}

}  // namespace Aperture
