#ifndef _USER_RADIATIVE_FUNCTIONS_H_
#define _USER_RADIATIVE_FUNCTIONS_H_

#include "cuda/constant_mem.h"
#include "cuda/cudarng.h"
#include "cuda/data_ptrs.h"
#include "cuda/radiation/rt_ic.h"
#include "cuda/radiation/rt_ic_dev.h"
#include "grids/grid_1dgr.h"
#include "radiation/spectra.h"
#include "sim_environment.h"

namespace Aperture {

namespace Kernels {

__device__ bool
check_emit_photon(data_ptrs& data, uint32_t tid, CudaRng& rng) {
  auto& ptc = data.particles;
  Scalar gamma = ptc.E[tid];
  float u = rng();
  return (u < find_ic_rate(gamma) * dev_params.delta_t);
}

__device__ void
emit_photon(data_ptrs& data, uint32_t tid, int offset, CudaRng& rng) {
  auto& ptc = data.particles;
  auto& photons = data.photons;

  Scalar x1 = ptc.x1[tid];
  Scalar p1 = ptc.p1[tid];
  uint32_t c = ptc.cell[tid];

  Scalar gamma = ptc.E[tid];
  Scalar Eph = gen_photon_e(gamma, &(rng.m_local_state));
  // Limit energy loss so that remaining particle momentum still
  // makes sense
  if (Eph >= gamma - 1.01f) Eph = gamma - 1.01f;

  ptc.E[tid] = (gamma - Eph);
   
  ptc.p1[tid] = ptc.E[tid] * p1 / gamma;
  // if p1 becomes NaN, set it to zero
  if (ptc.p1[tid] != ptc.p1[tid]) ptc.p1[tid] = 0.0f;

  // Scalar rate = find_gg_rate(Eph) * dev_params.delta_t;
  // float u = rng();
  // Scalar l_ph = -std::log(1.0 - u) / rate;

  // if (l_ph > dev_mesh.sizes[0]) return;
  // If photon energy is too low, do not track it, but still
  // subtract its energy as done above
  // if (std::abs(Eph) < dev_params.E_ph_min) return;
  if (std::abs(Eph) < 0.001f / dev_params.e_min) return;

  // Add the new photon
  // Scalar path = rad_model.draw_photon_freepath(Eph);
  // printf("Eph is %f, path is %f\n", Eph, path);
  photons.x1[offset] = x1;
  photons.p1[offset] = sgn(p1) * Eph;
  photons.E[offset] = Eph;
  photons.weight[offset] = ptc.weight[tid];
  photons.cell[offset] = c;
  // photons.path_left[offset] = 0.0;
  float u = rng();
  photons.flag[offset] =
      (u < dev_params.track_percent ? bit_or(PhotonFlag::tracked) : 0);
  if (u < dev_params.track_percent)
    photons.id[offset] = (dev_rank << 32) + atomicAdd(&dev_ph_id, 1);
}

__device__ bool
check_produce_pair(data_ptrs& data, uint32_t tid, CudaRng& rng) {
  auto& photons = data.photons;
  Scalar prob = find_gg_rate(photons.E[tid]) * dev_params.delta_t;
  return (rng() < prob);
  // return photons.path_left[tid] <= 0.0f;
}

__device__ void
produce_pair(data_ptrs& data, uint32_t tid, uint32_t offset,
             CudaRng& rng) {
  auto& ptc = data.particles;
  auto& photons = data.photons;

  Scalar Eph = std::abs(photons.p1[tid]);

  uint32_t c = photons.cell[tid];
  Pos_t x1 = photons.x1[tid];

  // Set this photon to be empty
  photons.cell[tid] = MAX_CELL;

  Scalar p1 = sgn(photons.p1[tid]) * std::sqrt(0.25f * Eph * Eph - 1.0f);
  if (p1 != p1) p1 = 0.0f;

  uint32_t offset_e = offset;
  uint32_t offset_p = offset + 1;

  // Add the two new particles
  ptc.x1[offset_e] = ptc.x1[offset_p] = x1;
  // printf("x1 = %f, x2 = %f, x3 = %f\n", ptc.x1[offset_e],
  // ptc.x2[offset_e], ptc.x3[offset_e]);

  Scalar beta = dev_mesh.pos(0, c, x1);
  ptc.p1[offset_e] = ptc.p1[offset_p] = p1;
  ptc.E[offset_e] = ptc.E[offset_p] = std::sqrt(1.0f + p1 * p1 + beta * beta);

#ifndef NDEBUG
  assert(ptc.cell[offset_e] == MAX_CELL);
  assert(ptc.cell[offset_p] == MAX_CELL);
#endif

  ptc.weight[offset_e] = ptc.weight[offset_p] = photons.weight[tid];
  ptc.cell[offset_e] = ptc.cell[offset_p] = c;
  float u = rng();
  if (u < dev_params.track_percent) {
    ptc.id[offset_e] = (dev_rank << 32) + atomicAdd(&dev_ptc_id, 1);
    ptc.id[offset_p] = (dev_rank << 32) + atomicAdd(&dev_ptc_id, 1);
    ptc.flag[offset_e] = set_ptc_type_flag(
        bit_or(ParticleFlag::secondary, ParticleFlag::tracked), ParticleType::electron);
    ptc.flag[offset_p] = set_ptc_type_flag(
        bit_or(ParticleFlag::secondary, ParticleFlag::tracked), ParticleType::positron);
  } else {
    ptc.flag[offset_e] = set_ptc_type_flag(
        bit_or(ParticleFlag::secondary), ParticleType::electron);
    ptc.flag[offset_p] = set_ptc_type_flag(
        bit_or(ParticleFlag::secondary), ParticleType::positron);
  }

}

}  // namespace Kernels

void
user_rt_init(sim_environment& env) {
  static inverse_compton rt_ic(env.params());
  Logger::print_debug("in rt_init, emin is {}", env.params().e_min);
  static Spectra::broken_power_law rt_ne(1.25, 1.1, env.params().e_min,
                                         1.0e-10, 0.1);
  rt_ic.init(rt_ne, rt_ne.emin(), rt_ne.emax(),
             1.50e24 / env.params().ic_path);
}

}  // namespace Aperture

#endif  // _USER_RADIATIVE_FUNCTIONS_H_
