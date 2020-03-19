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
  bool emit = check_bit(ptc.flag[tid], ParticleFlag::emit_photon);

  if (emit) {
    ptc.flag[tid] &= ~bit_or(ParticleFlag::emit_photon);
  }
  return emit;
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
  Scalar r = exp(dev_mesh.pos(0, c1, x1));
  Scalar theta = dev_mesh.pos(1, c2, x2);
  Scalar gamma = ptc.E[tid];
  Scalar pi = std::sqrt(p1 * p1 + p2 * p2 + p3 * p3);
  // Scalar u = rng();

  Interpolator2D<Spline::spline_t<1>> interp;
  Scalar B1 = interp(data.B1, x1, x2, c1, c2, Stagger(0b001));
  Scalar B2 = interp(data.B2, x1, x2, c1, c2, Stagger(0b010));
  Scalar B3 = interp(data.B3, x1, x2, c1, c2, Stagger(0b100));
  Scalar B = sqrt(B1 * B1 + B2 * B2 + B3 * B3);
  Scalar pdotB = (p1 * B1 + p2 * B2 + p3 * B3);
  Scalar p_mag_signed = sgn(pdotB) * sgn(B1) * std::abs(pdotB) / B;
  Scalar g = sqrt(1.0f + p_mag_signed * p_mag_signed);
  Scalar mu = std::abs(B1 / B);
  Scalar y = (B / dev_params.BQ) /
             (dev_params.star_kT * (g - p_mag_signed * mu));
  if (y < 20.0f && y > 0.0f) {
    Scalar coef = dev_params.res_drag_coef * y * y * y /
                  (r * r * (std::exp(y) - 1.0f));
    Scalar Ndot = std::abs(coef * (1.0f - p_mag_signed * mu / g));
    float theta_p = CONST_PI * rng();
    float u = cos(theta_p);

    // The abs is not necessary?
    Scalar Eph = std::abs(
        gamma * (g - std::abs(p_mag_signed) * u) *
        (1.0f - 1.0f / std::sqrt(1.0f + 2.0f * B / dev_params.BQ)));
    if (Eph < 2.1f) {
      // Treat this as hard X-ray emission
        Scalar angle =
            acos(sgn(pdotB) * (B1 * cos(theta) - B2 * sin(theta)) / B);
        float phi_p = 2.0f * CONST_PI * rng();
        Scalar cos_angle =
            std::cos(angle) * std::cos(theta_p) +
            std::sin(angle) * std::sin(theta_p) * std::cos(phi_p);
        angle = std::acos(cos_angle);
        auto& ph_flux = data.ph_flux;
        if (p1 > 0.0f && gamma > 1.5f) {
          Eph = std::log(std::abs(Eph)) / std::log(10.0f);
          if (Eph < -6.0f) Eph = -6.0f;
          int n0 = ((Eph + 6.0f) / 8.1f * (ph_flux.p.xsize - 1));
          if (n0 < 0) n0 = 0;
          if (n0 >= ph_flux.p.xsize) n0 = ph_flux.p.xsize - 1;
          int n1 = (std::abs(angle) / (CONST_PI + 1.0e-5)) *
                   (ph_flux.p.ysize - 1);
          if (n1 < 0) n1 = 0;
          if (n1 >= ph_flux.p.ysize) n1 = ph_flux.p.ysize - 1;
          auto w = ptc.weight[tid];
          atomicAdd(&ph_flux(n0, n1), Ndot * dev_params.delta_t * w);
          // printf("n0 is %d, n1 is %d, Ndot is %f, ph_flux is %f\n",
          // n0,
          //        n1, Ndot, ph_flux(n0, n1));
        }
      Scalar pf = std::sqrt(square(max(gamma - Eph * Ndot, 1.1f)) - 1.0f);
      ptc.p1[tid] = p1 * pf / pi;
      ptc.p2[tid] = p2 * pf / pi;
      ptc.p3[tid] = p3 * pf / pi;
      ptc.E[tid] = std::sqrt(1.0 + ptc.p1[tid] * ptc.p1[tid] +
                             ptc.p2[tid] * ptc.p2[tid] +
                             ptc.p3[tid] * ptc.p3[tid]);
    } else {
      // Treat this as a discrete photon emission
      if (Eph > gamma - 1.0f) Eph = gamma - 1.1f;
      float v = rng();
      if (v < Ndot * dev_params.delta_t) {
        Scalar pf = std::sqrt(square(gamma - Eph) - 1.0f);
        // gamma = (gamma - std::abs(Eph));
        ptc.p1[tid] = p1 * pf / pi;
        ptc.p2[tid] = p2 * pf / pi;
        ptc.p3[tid] = p3 * pf / pi;
        ptc.E[tid] = std::sqrt(1.0 + ptc.p1[tid] * ptc.p1[tid] +
                               ptc.p2[tid] * ptc.p2[tid] +
                               ptc.p3[tid] * ptc.p3[tid]);
        if (ptc.E[tid] != ptc.E[tid]) {
          printf(
              "NaN detected in photon emission! p1 is %f, p2 is %f, p3 "
              "is "
              "%f, gamma "
              "is %f\n",
              p1, p2, p3, gamma);
          asm("trap;");
          // p1 = p2 = p3 = 0.0f;
        }

        photons.x1[offset] = ptc.x1[tid];
        photons.x2[offset] = ptc.x2[tid];
        photons.x3[offset] = ptc.x3[tid];
        photons.p1[offset] = Eph * p1 / pi;
        photons.p2[offset] = Eph * p2 / pi;
        photons.p3[offset] = Eph * p3 / pi;
        photons.E[offset] = Eph;
        photons.weight[offset] = ptc.weight[tid];
        photons.path_left[offset] = dev_params.photon_path;
        photons.cell[offset] = ptc.cell[tid];
      }
    }
  }
}

__device__ bool
check_produce_pair(data_ptrs& data, uint32_t tid, CudaRng& rng) {
  auto& photons = data.photons;
  uint32_t cell = photons.cell[tid];
  int c1 = dev_mesh.get_c1(cell);
  int c2 = dev_mesh.get_c2(cell);
  // auto x1 = data.photons.x1[tid];
  auto x2 = data.photons.x2[tid];
  // auto p1 = data.photons.p1[tid];
  // auto p2 = data.photons.p2[tid];
  // auto p3 = data.photons.p3[tid];
  // auto Eph = data.photons.E[tid];
  Scalar theta = dev_mesh.pos(1, c2, x2);
  Scalar r = exp(dev_mesh.pos(0, c1, x1));
  // Do not care about photons in the first and last theta cell
  if (theta < dev_mesh.delta[1] ||
      theta > CONST_PI - dev_mesh.delta[1]) {
    photons.cell[tid] = MAX_CELL;
    return false;
  }

  // Scalar rho = max(
  //     std::abs(data.Rho[0](c1, c2) + data.Rho[1](c1, c2)),
  //     0.0001f);
  Scalar N = dev_params.q_e * std::abs(data.Rho[0](c1, c2)) + std::abs(data.Rho[1](c1, c2));
  // Scalar multiplicity = N / rho;
  // if (multiplicity > 100.0f) {
  if (N > 2.0f * square(1.0f / dev_mesh.delta[1] / r) * sin(theta)) {
    // Multiplicity already too high, kill photon but do not make a pair
    photons.cell[tid] = MAX_CELL;
    return false;
  }
  return (photons.path_left[tid] <= 0.0f);
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
