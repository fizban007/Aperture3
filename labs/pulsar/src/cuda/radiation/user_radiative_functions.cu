#ifndef _USER_RADIATIVE_FUNCTIONS_H_
#define _USER_RADIATIVE_FUNCTIONS_H_

#include "cuda/constant_mem.h"
#include "cuda/cudarng.h"
#include "cuda/data_ptrs.h"
#include "cuda/grids/grid_log_sph_ptrs.h"
#include "cuda/utils/interpolation.cuh"
#include "grids/grid_log_sph.h"
#include "sim_environment.h"

namespace Aperture {

namespace Kernels {

__device__ mesh_ptrs_log_sph dev_mesh_ptrs_log_sph;

__device__ Scalar rho_c(Scalar r, Scalar theta) {
  Scalar c = std::cos(theta);
  Scalar c2 = c * c;
  Scalar s = std::sin(theta);
  if (std::abs(s) < 1.0e-5) return 1.0e10;
  return r * std::pow(1.0f + 3.0f * c2, 1.5) / (3.0f * s * (1.0f + c2));
}

__device__ bool
check_emit_photon(data_ptrs& data, uint32_t tid, CudaRng& rng) {
  auto& ptc = data.particles;
  auto c = ptc.cell[tid];
  auto c1 = dev_mesh.get_c1(c);
  auto c2 = dev_mesh.get_c2(c);
  Scalar r = std::exp(dev_mesh.pos(0, c1, ptc.x1[tid]));
  Scalar gamma = ptc.E[tid];
  // Skip photon emission when outside given radius (light cylinder)
  if (r > 0.6f / dev_params.omega) return false;
  if (r < 1.05) return false;
  Scalar theta = dev_mesh.pos(1, c2, ptc.x2[tid]);

  // if (gamma > dev_params.gamma_thr)
  // //   printf("emitted a photon at cell %d, %d, gamma is %f\n", c1, c2,
  // //   gamma);
  // // Scalar gamma = data.particles.E[tid];
  // return (gamma > dev_params.gamma_thr && r < dev_params.r_cutoff &&
  //         r > 1.02f);
  float u = rng();
  float rc = rho_c(r, theta);
  if (rc < 1.0f) return false;
  float rate = dev_params.delta_t * gamma / (dev_params.l_curv * rc);
  float Eph = dev_params.e_curv * gamma * gamma * gamma / rc;
  // if (u <= rate && Eph > 1.0f) {
  //   printf("r is %f, theta is %f, gamma is %f, rho_c is %f, Eph is %f\n", r, theta, gamma, rc, Eph);
  // }
  return (u < rate && Eph > 1.0f);
}

__device__ void
emit_photon(data_ptrs& data, uint32_t tid, int offset, CudaRng& rng) {
  auto& ptc = data.particles;
  auto& photons = data.photons;

  auto c = ptc.cell[tid];
  Scalar p1 = ptc.p1[tid];
  Scalar p2 = ptc.p2[tid];
  Scalar p3 = ptc.p3[tid];
  auto x1 = ptc.x1[tid];
  auto x2 = ptc.x2[tid];
  auto c1 = dev_mesh.get_c1(c);
  auto c2 = dev_mesh.get_c2(c);
  // Scalar gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
  Scalar gamma = ptc.E[tid];
  // Scalar pi = std::sqrt(gamma * gamma - 1.0f);
  Scalar pi = std::sqrt(p1 * p1 + p2 * p2 + p3 * p3);
  Scalar r = std::exp(dev_mesh.pos(0, c1, x1));
  Scalar theta = dev_mesh.pos(1, c2, x2);
  // Scalar u = rng();

  // Energy of the photon emitted
  // Scalar Eph = 2.5f + u * (dev_params.E_secondary - 1.0f) * 2.0f;
  Scalar Eph = std::min(gamma - 1.0f, dev_params.e_curv * gamma * gamma * gamma / rho_c(r, theta));
  // if (tid < 1000) {
  //   printf("gamma is %f, Eph is %f\n", gamma, Eph);
  // }
  // if (Eph < 2.1f) return;

  Scalar pf = std::sqrt(square(gamma - Eph) - 1.0f);
  // gamma = (gamma - std::abs(Eph));
  ptc.p1[tid] = p1 * pf / pi;
  ptc.p2[tid] = p2 * pf / pi;
  ptc.p3[tid] = p3 * pf / pi;
  ptc.E[tid] = gamma - Eph;

  Interpolator2D<Spline::spline_t<1>> interp;
  Scalar B1 = interp(data.B1, x1, x2, c1, c2, Stagger(0b001));
  Scalar B2 = interp(data.B2, x1, x2, c1, c2, Stagger(0b001));
  Scalar B3 = interp(data.B3, x1, x2, c1, c2, Stagger(0b001));
  Scalar B = sqrt(B1 * B1 + B2 * B2 + B3 * B3);

  if (Eph < 2.1f) return;
  if (Eph * B / dev_params.BQ < 2.1f && p1 > 0.0f) return;
  // Do not track photons that can't convert
  // Scalar lph = min(
  //     10.0f, (1.0f / std::sin(theta) - 1.0f) * dev_params.photon_path);
  // If photon energy is too low, do not track it, but still
  // subtract its energy as done above
  // if (std::abs(Eph) < dev_params.E_ph_min) return;
  // if (theta < 0.265f || theta > CONST_PI - 0.265f) return;
  // if (theta < 0.165f || theta > CONST_PI - 0.165f) return;
  // if (theta < 0.005f || theta > CONST_PI - 0.005f) return;

  // u = rng();
  // Add the new photon
  // Scalar path = rad_model.draw_photon_freepath(Eph);
  // Scalar path =
  //     dev_params.photon_path * std::sqrt(-2.0f * std::log(u));
  // Scalar path = lph * std::sqrt(-2.0f * std::log(u));
  // Scalar path = lph * (0.5f + 0.5f * u);
  // if (path > dev_params.r_cutoff) return;
  // Scalar path = dev_params.photon_path;
  // if (path > dev_params.lph_cutoff) return;
  // printf("Eph is %f, path is %f\n", Eph, path);
  photons.x1[offset] = ptc.x1[tid];
  photons.x2[offset] = ptc.x2[tid];
  photons.x3[offset] = ptc.x3[tid];
  photons.p1[offset] = Eph * p1 / pi;
  photons.p2[offset] = Eph * p2 / pi;
  photons.p3[offset] = Eph * p3 / pi;
  photons.E[offset] = Eph;
  photons.weight[offset] = ptc.weight[tid];
  photons.path_left[offset] = 0.0;
  photons.cell[offset] = ptc.cell[tid];

}

__device__ bool
check_produce_pair(data_ptrs& data, uint32_t tid, CudaRng& rng) {
  auto& photons = data.photons;
  uint32_t cell = photons.cell[tid];
  int c1 = dev_mesh.get_c1(cell);
  int c2 = dev_mesh.get_c2(cell);
  auto x1 = data.photons.x1[tid];
  auto x2 = data.photons.x2[tid];
  auto p1 = data.photons.p1[tid];
  auto p2 = data.photons.p2[tid];
  auto p3 = data.photons.p3[tid];
  auto Eph = data.photons.E[tid];
  Scalar theta = dev_mesh.pos(1, c2, x2);
  // Do not care about photons in the first and last theta cell
  if (theta < dev_mesh.delta[1] ||
      theta > CONST_PI - dev_mesh.delta[1]) {
    photons.cell[tid] = MAX_CELL;
    return false;
  }

  Scalar rho = max(
      std::abs(data.Rho[0](c1, c2) + data.Rho[1](c1, c2)),
      0.0001f);
  Scalar N = std::abs(data.Rho[0](c1, c2)) + std::abs(data.Rho[1](c1, c2));
  Scalar multiplicity = N / rho;
  if (multiplicity > 50.0f) {
    photons.cell[tid] = MAX_CELL;
    return false;
  }
  // return (photons.path_left[tid] <= 0.0f);
  Interpolator2D<Spline::spline_t<1>> interp;
  Scalar B1 = interp(data.B1, x1, x2, c1, c2, Stagger(0b001));
  Scalar B2 = interp(data.B2, x1, x2, c1, c2, Stagger(0b001));
  Scalar B3 = interp(data.B3, x1, x2, c1, c2, Stagger(0b001));
  Scalar B = sqrt(B1 * B1 + B2 * B2 + B3 * B3);
  Scalar cth = (B1 * p1 + B2 * p2 + B3 * p3) / (B * Eph);
  Scalar chi = Eph * B / dev_params.BQ * sqrt(1.0 - cth * cth);
  return chi > 0.12;
}

__device__ void
produce_pair(data_ptrs& data, uint32_t tid, uint32_t offset,
             CudaRng& rng) {
  auto& ptc = data.particles;
  auto& photons = data.photons;

  Scalar p1 = photons.p1[tid];
  Scalar p2 = photons.p2[tid];
  Scalar p3 = photons.p3[tid];
  Scalar E_ph2 = p1 * p1 + p2 * p2 + p3 * p3;
  if (E_ph2 <= 4.01f) E_ph2 = 4.01f;

  Scalar ratio = std::sqrt(0.25f - 1.0f / E_ph2);
  Scalar gamma = sqrt(1.0f + ratio * ratio * E_ph2);

  if (gamma != gamma) {
    photons.cell[tid] = MAX_CELL;
    return;
  }
  // Add the two new particles
  int offset_e = offset;
  int offset_p = offset + 1;
  // int offset_p = ptc_num + start_pos + pos_in_block +
  // pair_count[blockIdx.x];

  ptc.x1[offset_e] = ptc.x1[offset_p] = photons.x1[tid];
  ptc.x2[offset_e] = ptc.x2[offset_p] = photons.x2[tid];
  ptc.x3[offset_e] = ptc.x3[offset_p] = photons.x3[tid];
  // printf("x1 = %f, x2 = %f, x3 = %f\n", ptc.x1[offset_e],
  // ptc.x2[offset_e], ptc.x3[offset_e]);

  ptc.p1[offset_e] = ptc.p1[offset_p] = ratio * p1;
  ptc.p2[offset_e] = ptc.p2[offset_p] = ratio * p2;
  ptc.p3[offset_e] = ptc.p3[offset_p] = ratio * p3;
  ptc.E[offset_e] = ptc.E[offset_p] = gamma;

#ifndef NDEBUG
  assert(ptc.cell[offset_e] == MAX_CELL);
  assert(ptc.cell[offset_p] == MAX_CELL);
#endif

  // float u = rng();
  ptc.weight[offset_e] = ptc.weight[offset_p] = photons.weight[tid];
  ptc.cell[offset_e] = ptc.cell[offset_p] = photons.cell[tid];
  ptc.flag[offset_e] = set_ptc_type_flag(
           bit_or(ParticleFlag::secondary), ParticleType::electron);
  ptc.flag[offset_p] = set_ptc_type_flag(
           bit_or(ParticleFlag::secondary), ParticleType::positron);

  // Set this photon to be empty
  photons.cell[tid] = MAX_CELL;
}

}  // namespace Kernels

void
user_rt_init(sim_environment& env) {
  // Copy the mesh pointer to device memory
  Grid_LogSph* grid = dynamic_cast<Grid_LogSph*>(&env.local_grid());
  auto ptrs = get_mesh_ptrs(*grid);
  CudaSafeCall(cudaMemcpyToSymbol(Kernels::dev_mesh_ptrs_log_sph,
                                  (void*)&ptrs,
                                  sizeof(mesh_ptrs_log_sph)));
}

}  // namespace Aperture

#endif  // _USER_RADIATIVE_FUNCTIONS_H_
