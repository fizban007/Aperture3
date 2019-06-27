#ifndef _USER_RADIATIVE_FUNCTIONS_H_
#define _USER_RADIATIVE_FUNCTIONS_H_

#include "cuda/constant_mem.h"
#include "cuda/cudarng.h"
#include "cuda/data_ptrs.h"

namespace Aperture {

namespace Kernels {

__device__ bool
check_emit_photon(data_ptrs& data, uint32_t tid, CudaRng& rng) {
  Scalar gamma = data.particles.E[tid];
  return (gamma > dev_params.gamma_thr);
}

__device__ void
emit_photon(data_ptrs& data, uint32_t tid, int offset, CudaRng& rng) {
  auto& ptc = data.particles;
  auto& photons = data.photons;

  Scalar p1 = ptc.p1[tid];
  Scalar p2 = ptc.p2[tid];
  Scalar p3 = ptc.p3[tid];
  //   // Scalar gamma = sqrt(1.0f + p1 * p1 + p2 * p2 + p3 * p3);
  Scalar gamma = ptc.E[tid];
  Scalar pi = std::sqrt(gamma * gamma - 1.0f);

  Scalar u = rng();
  Scalar Eph = 2.5f + u * (dev_params.E_secondary - 1.0f) * 2.0f;
  Scalar pf = std::sqrt(square(gamma - Eph) - 1.0f);

  ptc.p1[tid] = p1 * pf / pi;
  ptc.p2[tid] = p2 * pf / pi;
  ptc.p3[tid] = p3 * pf / pi;
  ptc.E[tid] = gamma - Eph;

  auto c = ptc.cell[tid];
  Scalar theta = dev_mesh.pos(1, dev_mesh.get_c2(c), ptc.x2[tid]);
  Scalar lph = min(
      10.0f, (1.0f / std::sin(theta) - 1.0f) * dev_params.photon_path);
  // If photon energy is too low, do not track it, but still
  // subtract its energy as done above
  // if (std::abs(Eph) < dev_params.E_ph_min) continue;
  if (theta < 0.005f || theta > CONST_PI - 0.005f) return;

  u = rng();
  // Add the new photo
  Scalar path = lph * (0.5f + 0.5f * u);
  if (path > dev_params.r_cutoff) return;
  // printf("Eph is %f, path is %f\n", Eph, path);
  photons.x1[offset] = ptc.x1[tid];
  photons.x2[offset] = ptc.x2[tid];
  photons.x3[offset] = ptc.x3[tid];
  photons.p1[offset] = Eph * p1 / pi;
  photons.p2[offset] = Eph * p2 / pi;
  photons.p3[offset] = Eph * p3 / pi;
  photons.weight[offset] = ptc.weight[tid];
  photons.path_left[offset] = path;
  photons.cell[offset] = ptc.cell[tid];
}

__device__ bool
check_produce_pair(data_ptrs& data, uint32_t tid, CudaRng& rng) {
  return data.photons.path_left[tid] <= 0.0f;
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
  // Scalar new_p = std::sqrt(max(0.25f * E_ph *
  // E_ph, 1.0f)
  // - 1.0f);
  Scalar ratio = std::sqrt(0.25f - 1.0f / E_ph2);
  Scalar gamma = sqrt(1.0f + ratio * ratio * E_ph2);
  if (gamma != gamma) {
    // printf(
    //     "NaN detected in pair creation! ratio is %f, // E_ph2 is %f,
    //     " "p1 is "
    //     "%f, "
    //     "p2 is %f, p3 is %f\n",
    //     ratio, E_ph2, p1, p2, p3);
    // asm("trap;");
    photons.cell[tid] = MAX_CELL;
    return;
  }
  uint32_t offset_e = offset;
  uint32_t offset_p = offset + 1;

  ptc.x1[offset_e] = ptc.x1[offset_p] = photons.x1[tid];
  ptc.x2[offset_e] = ptc.x2[offset_p] = photons.x2[tid];
  ptc.x3[offset_e] = ptc.x3[offset_p] = photons.x3[tid];
  // printf("x1 = %f, x2 = %f, x3 = %f\n",
  // ptc.x1[offset_e],
  // ptc.x2[offset_e], ptc.x3[offset_e]);

  ptc.p1[offset_e] = ptc.p1[offset_p] = ratio * p1;
  ptc.p2[offset_e] = ptc.p2[offset_p] = ratio * p2;
  ptc.p3[offset_e] = ptc.p3[offset_p] = ratio * p3;
  ptc.E[offset_e] = ptc.E[offset_p] = gamma;

#ifndef NDEBUG
  assert(ptc.cell[offset_e] == MAX_CELL);
  assert(ptc.cell[offset_p] == MAX_CELL);
#endif
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

}  // namespace Aperture

#endif  // _USER_RADIATIVE_FUNCTIONS_H_
