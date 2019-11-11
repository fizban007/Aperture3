#ifndef _USER_RADIATIVE_FUNCTIONS_H_
#define _USER_RADIATIVE_FUNCTIONS_H_

#include "cuda/constant_mem.h"
#include "cuda/cudarng.h"
#include "cuda/data_ptrs.h"
#include "cuda/grids/grid_log_sph_ptrs.h"
#include "grids/grid_log_sph.h"
#include "sim_environment.h"

namespace Aperture {

namespace Kernels {

__device__ mesh_ptrs_log_sph dev_mesh_ptrs_log_sph;

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
  int c1 = dev_mesh.get_c1(c);
  int c2 = dev_mesh.get_c2(c);

  Scalar B1 = data.B1(c1, c2) + data.Bbg1(c1, c2);
  Scalar B2 = data.B2(c1, c2) + data.Bbg2(c1, c2);
  Scalar B3 = data.B3(c1, c2);
  Scalar b = std::sqrt(B1 * B1 + B2 * B2 + B3 * B3) / dev_params.BQ;
  // float u = rng();

  Scalar p1 = ptc.p1[tid];
  Scalar p2 = ptc.p2[tid];
  Scalar p3 = ptc.p3[tid];
  Scalar gamma = ptc.E[tid];
  Scalar pi = std::sqrt(gamma * gamma - 1.0f);
  // Scalar Eph = (gamma - pi * (2.0f * u - 1.0f)) *
  Scalar Eph = gamma *
               (1.0f - 1.0f / std::sqrt(1.0f + 2.0f * b));
  if (Eph > gamma - 1.0f) Eph = gamma - 1.1f;
  Scalar pf = std::sqrt(square(gamma - Eph) - 1.0f);
  // gamma = (gamma - std::abs(Eph));
  ptc.p1[tid] = p1 * pf / pi;
  ptc.p2[tid] = p2 * pf / pi;
  ptc.p3[tid] = p3 * pf / pi;
  ptc.E[tid] = gamma - Eph;

  // If photon energy is too low, do not track it, but still
  // subtract its energy as done above
  if (Eph < dev_params.E_ph_min) return;

  photons.x1[offset] = ptc.x1[tid];
  photons.x2[offset] = ptc.x2[tid];
  photons.x3[offset] = ptc.x3[tid];
  photons.p1[offset] = Eph * p1 / pi;
  photons.p2[offset] = Eph * p2 / pi;
  photons.p3[offset] = Eph * p3 / pi;
  photons.weight[offset] = ptc.weight[tid];
  photons.path_left[offset] = dev_params.photon_path;
  photons.cell[offset] = ptc.cell[tid];

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
  int c1 = dev_mesh.get_c1(cell);
  int c2 = dev_mesh.get_c2(cell);
  Scalar theta = dev_mesh.pos(1, c2, photons.x2[tid]);
  if (theta < dev_mesh.delta[1] * 15 ||
      theta > CONST_PI - dev_mesh.delta[1] * 4) {
    photons.cell[tid] = MAX_CELL;
    return false;
  }
  Scalar rho = max(
      std::abs(data.Rho[0](c1, c2) + data.Rho[1](c1, c2)),
      0.0001f);
  Scalar N = std::abs(data.Rho[0](c1, c2)) + std::abs(data.Rho[1](c1, c2));
  Scalar multiplicity = N / rho;
  if (multiplicity > 20.0f) {
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
  // TODO: track photons
  ptc.weight[offset_e] = ptc.weight[offset_p] = photons.weight[tid];
  ptc.cell[offset_e] = ptc.cell[offset_p] = photons.cell[tid];
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
