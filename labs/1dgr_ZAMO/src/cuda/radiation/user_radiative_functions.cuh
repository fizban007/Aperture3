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

__device__ Grid_1dGR::mesh_ptrs dev_mesh_ptrs_1dgr;

__device__ bool
check_emit_photon(data_ptrs& data, uint32_t tid, CudaRng& rng) {
  auto& ptc = data.particles;
  uint32_t cell = ptc.cell[tid];
  auto x1 = ptc.x1[tid];
  Scalar alpha = dev_mesh_ptrs_1dgr.alpha[cell] * x1 +
                 dev_mesh_ptrs_1dgr.alpha[cell - 1] * (1.0f - x1);
  Scalar gamma = alpha * ptc.E[tid];
  float u = rng();
  return (u < find_ic_rate(gamma) * alpha * dev_params.delta_t);
}

__device__ void
emit_photon(data_ptrs& data, uint32_t tid, int offset, CudaRng& rng) {
  auto& ptc = data.particles;
  auto& photons = data.photons;

  Scalar x1 = ptc.x1[tid];
  Scalar p1 = ptc.p1[tid];
  Scalar u0_ptc = ptc.E[tid];
  uint32_t c = ptc.cell[tid];
  Scalar xi = dev_mesh.pos(0, c, x1);

  const Scalar a = dev_params.a;
  // const Scalar a = 0.0;
  const Scalar rp = 1.0f + std::sqrt(1.0f - a * a);
  const Scalar rm = 1.0f - std::sqrt(1.0f - a * a);
  Scalar exp_xi = std::exp(xi * (rp - rm));
  Scalar r = (rp - rm * exp_xi) / (1.0 - exp_xi);
  Scalar Delta = r * r - 2.0 * r + a * a;

  Scalar alpha = dev_mesh_ptrs_1dgr.alpha[c] * x1 +
                 dev_mesh_ptrs_1dgr.alpha[c - 1] * (1.0f - x1);
  Scalar D1 = dev_mesh_ptrs_1dgr.D1[c] * x1 +
              dev_mesh_ptrs_1dgr.D1[c - 1] * (1.0f - x1);
  Scalar D2 = dev_mesh_ptrs_1dgr.D2[c] * x1 +
              dev_mesh_ptrs_1dgr.D2[c - 1] * (1.0f - x1);
  Scalar D3 = dev_mesh_ptrs_1dgr.D3[c] * x1 +
              dev_mesh_ptrs_1dgr.D3[c - 1] * (1.0f - x1);
  Scalar B3B1 = dev_mesh_ptrs_1dgr.B3B1[c] * x1 +
                dev_mesh_ptrs_1dgr.B3B1[c - 1] * (1.0f - x1);
  Scalar g11 = dev_mesh_ptrs_1dgr.gamma_rr[c] * x1 +
               dev_mesh_ptrs_1dgr.gamma_rr[c - 1] * (1.0f - x1);
  Scalar g33 = dev_mesh_ptrs_1dgr.gamma_ff[c] * x1 +
               dev_mesh_ptrs_1dgr.gamma_ff[c - 1] * (1.0f - x1);
  Scalar beta = dev_mesh_ptrs_1dgr.beta_phi[c] * x1 +
                dev_mesh_ptrs_1dgr.beta_phi[c - 1] * (1.0f - x1);

  Scalar ur_ptc = u0_ptc * Delta * (p1 / u0_ptc - D1) / D2;
  // Scalar uphi_ptc = dev_params.omega * u0_ptc + B3B1 * ur_ptc;

  Scalar gamma = alpha * ptc.E[tid];
  Scalar Eph = gen_photon_e(gamma, &(rng.m_local_state));
  // Scalar Eph = bb * dev_ic_dep * gamma;
  // Limit energy loss so that remaining particle momentum still
  // makes sense
  // if (Eph >= gamma - 1.01f) Eph = gamma - 1.01f;

  ptc.E[tid] = (gamma - Eph) / alpha;

  ptc.p1[tid] =
      sgn(p1) * std::sqrt(square(ptc.E[tid]) *
                              (D2 * (alpha * alpha - D3) + D1 * D1) -
                          D2);
  // if p1 becomes NaN, set it to zero
  if (ptc.p1[tid] != ptc.p1[tid]) ptc.p1[tid] = 0.0f;

  // If photon energy is too low, do not track it, but still
  // subtract its energy as done above
  // if (std::abs(Eph) < dev_params.E_ph_min) return;
  // if (std::abs(Eph) < 0.01f / dev_params.e_min) return;
  if (std::abs(Eph) < 50.0f) return;

  // Add the new photon
  // Scalar path = rad_model.draw_photon_freepath(Eph);
  // printf("Eph is %f, path is %f\n", Eph, path);
  photons.x1[offset] = ptc.x1[tid];
  // photons.p1[offset] = Delta * (Eph / gamma) * ur_ptc / g11;
  // photons.p3[offset] =
  //     (Eph / gamma) *
  //     (u0_ptc * (dev_params.omega + beta) + B3B1 * ur_ptc) / g33;
  photons.p1[offset] = Delta * (Eph / std::sqrt(gamma * gamma - 1.0)) * ur_ptc / g11;
  photons.p3[offset] =
      (Eph / std::sqrt(gamma * gamma - 1.0)) *
      (u0_ptc * (dev_params.omega + beta) + B3B1 * ur_ptc) / g33;
  photons.E[offset] = Eph / alpha;
  photons.weight[offset] = ptc.weight[tid];
  photons.cell[offset] = c;
  float u = rng();
  photons.flag[offset] =
      (u < dev_params.track_percent ? bit_or(PhotonFlag::tracked) : 0);
  if (u < dev_params.track_percent)
    photons.id[offset] = (dev_rank << 32) + atomicAdd(&dev_ph_id, 1);
}

__device__ bool
check_produce_pair(data_ptrs& data, uint32_t tid, CudaRng& rng) {
  auto& photons = data.photons;
  uint32_t cell = photons.cell[tid];

  auto x1 = photons.x1[tid];

  Scalar alpha = dev_mesh_ptrs_1dgr.alpha[cell] * x1 +
                 dev_mesh_ptrs_1dgr.alpha[cell - 1] * (1.0f - x1);
  Scalar u0_hat = alpha * std::abs(photons.E[tid]);
  // if (u0_hat < dev_params.E_ph_min) {
  //   photons.cell[tid] = MAX_CELL;
  //   return false;
  // }

  Scalar prob = find_gg_rate(u0_hat) * alpha * dev_params.delta_t;
  float u = rng();
  return u < prob;
}

__device__ void
produce_pair(data_ptrs& data, uint32_t tid, uint32_t offset,
             CudaRng& rng) {
  auto& ptc = data.particles;
  auto& photons = data.photons;

  uint32_t c = photons.cell[tid];
  Pos_t x1 = photons.x1[tid];
  Scalar alpha = dev_mesh_ptrs_1dgr.alpha[c] * x1 +
                 dev_mesh_ptrs_1dgr.alpha[c - 1] * (1.0f - x1);

  // Calculate energy of generated pairs
  Scalar u0;

  if (dev_params.rad_cooling_on) {
    const Scalar a = dev_params.a;
    // const Scalar a = 0.0;
    const Scalar rp = 1.0f + std::sqrt(1.0f - a * a);
    const Scalar rm = 1.0f - std::sqrt(1.0f - a * a);
    Scalar xi = dev_mesh.pos(0, c, x1);
    Scalar exp_xi = std::exp(xi * (rp - rm));
    Scalar r = (rp - rm * exp_xi) / (1.0 - exp_xi);
    Scalar Delta = r * r - 2.0 * r + a * a;
    Scalar theta = dev_mesh_ptrs_1dgr.theta[c] * x1 +
                   dev_mesh_ptrs_1dgr.theta[c - 1] * (1.0f - x1);
    Scalar A =
        square(r * r + a * a) - Delta * a * a * square(std::sin(theta));
    // upper components of gamma
    Scalar g11 = dev_mesh_ptrs_1dgr.gamma_rr[c] * x1 +
                 dev_mesh_ptrs_1dgr.gamma_rr[c - 1] * (1.0f - x1);
    Scalar g33 = dev_mesh_ptrs_1dgr.gamma_ff[c] * x1 +
                 dev_mesh_ptrs_1dgr.gamma_ff[c - 1] * (1.0f - x1);
    Scalar B3B1 = dev_mesh_ptrs_1dgr.B3B1[c] * x1 +
                  dev_mesh_ptrs_1dgr.B3B1[c - 1] * (1.0f - x1);
    // photon 4-velocity in ZAMO frame
    Scalar uph0 = alpha * photons.E[tid];
    // note that the following are the lower components
    Scalar uph1 = std::sqrt(g11) * photons.p1[tid] / Delta;
    Scalar uph3 = std::sqrt(g33) * photons.p3[tid];
    // 4-velocity of pairs in the ZAMO frame
    Scalar u10 = 0.5 * uph0;
    Scalar u11 = std::sqrt(u10 * u10 - 1.0) * uph1 / uph0;
    Scalar u13 = sgn(uph3) * std::sqrt(u10 * u10 - 1.0 - u11 * u11);
    // particle 4-velocity in corotating frame
    Scalar bb =
        (dev_params.omega - 2.0 * r * a / A) / alpha / std::sqrt(g33);
    if (bb > 1.0) bb = 1.0 - 1.0e-4;
    if (bb < -1.0) bb = -1.0 + 1.0e-4;
    Scalar gg = 1.0 / std::sqrt(1.0 - bb * bb);
    Scalar u23 = gg * (u13 - bb * u10);
    // angle between particle velocity and magnetic field
    Scalar b3 = gg * B3B1 * std::sqrt(g11) / std::sqrt(g33);
    Scalar cos_th = (u11 + u23 * b3) /
                    std::sqrt(u11 * u11 + u23 * u23) /
                    std::sqrt(1.0 + b3 * b3);
    if (cos_th > 1.0) cos_th = 1.0 - 1.0e-4;
    if (cos_th < -1.0) cos_th = -1.0 + 1.0e-4;
    Scalar sin_th = std::sqrt(1.0 - cos_th * cos_th);
    // particle 4-velocity in the corotating frame after cooling
    Scalar u30 = 1.0 / sin_th;
    Scalar u33 = std::sqrt(u30 * u30 - 1.0) * b3 /
                 std::sqrt(1.0 + b3 * b3);
    // in ZAMO frame
    Scalar u40 = gg * (u30 + bb * u33);
    if (u40 > u10 || u40 != u40) u40 = u10;
    // back to global coordinates
    u0 = u40 / alpha;
    if (u0 != u0)
      printf(
          "pair energy becomes NAN! bb is %f, u11 is %f, u13 is "
          "%f, uph0 is %f, uph1 is %f, cos_th is %f \n",
          bb, u11, u13, uph0, uph1, cos_th);
    // if (tid == 10) {
    if (true) {
      printf("Producing pair: r is %f, sin_th is %f, u10 is %f, u0 is %f\n", r,
             sin_th, u10, u0);
    }
  } else {
    u0 = 0.5f * std::abs(photons.E[tid]);
  }

  // Set this photon to be empty
  photons.cell[tid] = MAX_CELL;

  Scalar D1 = dev_mesh_ptrs_1dgr.D1[c] * x1 +
              dev_mesh_ptrs_1dgr.D1[c - 1] * (1.0f - x1);
  Scalar D2 = dev_mesh_ptrs_1dgr.D2[c] * x1 +
              dev_mesh_ptrs_1dgr.D2[c - 1] * (1.0f - x1);
  Scalar D3 = dev_mesh_ptrs_1dgr.D3[c] * x1 +
              dev_mesh_ptrs_1dgr.D3[c - 1] * (1.0f - x1);

  Scalar p1 =
      sgn(photons.p1[tid]) *
      std::sqrt(u0 * u0 * (D2 * (alpha * alpha - D3) + D1 * D1) - D2);
  if (p1 != p1) p1 = 0.0f;

  uint32_t offset_e = offset;
  uint32_t offset_p = offset + 1;

  // Add the two new particles
  ptc.x1[offset_e] = ptc.x1[offset_p] = x1;
  // printf("x1 = %f, x2 = %f, x3 = %f\n", ptc.x1[offset_e],
  // ptc.x2[offset_e], ptc.x3[offset_e]);

  ptc.p1[offset_e] = ptc.p1[offset_p] = p1;
  ptc.E[offset_e] = ptc.E[offset_p] = u0;

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
        bit_or(ParticleFlag::secondary, ParticleFlag::tracked),
        ParticleType::electron);
    ptc.flag[offset_p] = set_ptc_type_flag(
        bit_or(ParticleFlag::secondary, ParticleFlag::tracked),
        ParticleType::positron);
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
  // static Spectra::broken_power_law rt_ne(1.25, 1.1,
  // env.params().e_min,
  //                                        1.0e-10, 0.1);
  static Spectra::black_body rt_ne(env.params().e_min);
  rt_ic.init(rt_ne, rt_ne.emin(), rt_ne.emax(),
             1.50e24 / env.params().ic_path);

  // Copy the mesh pointer to device memory
  Grid_1dGR* grid = dynamic_cast<Grid_1dGR*>(&env.local_grid());
  auto ptrs = grid->get_mesh_ptrs();
  CudaSafeCall(cudaMemcpyToSymbol(Kernels::dev_mesh_ptrs_1dgr,
                                  (void*)&ptrs,
                                  sizeof(Grid_1dGR::mesh_ptrs)));
}

}  // namespace Aperture

#endif  // _USER_RADIATIVE_FUNCTIONS_H_