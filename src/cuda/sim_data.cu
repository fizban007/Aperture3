#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cudaUtility.h"
#include "cuda/cudarng.h"
#include "cuda/data_ptrs.h"
#include "cuda/kernels.h"
#include "cuda/utils/interpolation.cuh"
#include "cuda/utils/pitchptr.h"
#include "sim_data_impl.hpp"
#include "visit_struct/visit_struct.hpp"

namespace Aperture {

static data_ptrs g_ptrs;

namespace Kernels {

template <typename T>
__global__ void compute_EdotB_3d(pitchptr<T> e1, pitchptr<T> e2, pitchptr<T> e3,
                                 pitchptr<T> b1, pitchptr<T> b2, pitchptr<T> b3,
                                 pitchptr<T> EdotB) {
  // Compute time-averaged EdotB over the output interval
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x, c2 = threadIdx.y, c3 = threadIdx.z;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  int n3 = dev_mesh.guard[2] + t3 * blockDim.y + c3;

  size_t globalOffset = e1.compute_offset(n1, n2, n3);

  float delta = 1.0f / dev_params.data_interval;
  Scalar E1 = interpolate(e1, globalOffset, Stagger(0b110), Stagger(0b000),
                          e1.p.pitch, e1.p.ysize);
  Scalar E2 = interpolate(e2, globalOffset, Stagger(0b101), Stagger(0b000),
                          e2.p.pitch, e2.p.ysize);
  Scalar E3 = interpolate(e3, globalOffset, Stagger(0b011), Stagger(0b000),
                          e3.p.pitch, e3.p.ysize);
  Scalar B1 = interpolate(b1, globalOffset, Stagger(0b001), Stagger(0b000),
                          b1.p.pitch, b1.p.ysize);
  Scalar B2 = interpolate(b2, globalOffset, Stagger(0b010), Stagger(0b000),
                          b2.p.pitch, b2.p.ysize);
  Scalar B3 = interpolate(b3, globalOffset, Stagger(0b100), Stagger(0b000),
                          b3.p.pitch, b3.p.ysize);

  // Do the actual computation here
  EdotB[globalOffset] +=
      delta * (E1 * B1 + E2 * B2 + E3 * B3) / sqrt(B1 * B1 + B2 * B2 + B3 * B3);
}

template <typename T>
__global__ void compute_EdotB_2d(pitchptr<T> e1, pitchptr<T> e2, pitchptr<T> e3,
                                 pitchptr<T> b1, pitchptr<T> b2, pitchptr<T> b3,
                                 pitchptr<T> EdotB) {
  // Compute time-averaged EdotB over the output interval
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;

  size_t globalOffset = e1.compute_offset(n1, n2);

  float delta = 1.0f / dev_params.data_interval;
  Scalar E1 = interpolate2d(e1, globalOffset, Stagger(0b110), Stagger(0b000),
                            e1.p.pitch);
  Scalar E2 = interpolate2d(e2, globalOffset, Stagger(0b101), Stagger(0b000),
                            e2.p.pitch);
  Scalar E3 = interpolate2d(e3, globalOffset, Stagger(0b011), Stagger(0b000),
                            e3.p.pitch);
  Scalar B1 = interpolate2d(b1, globalOffset, Stagger(0b001), Stagger(0b000),
                            b1.p.pitch);
  Scalar B2 = interpolate2d(b2, globalOffset, Stagger(0b010), Stagger(0b000),
                            b2.p.pitch);
  Scalar B3 = interpolate2d(b3, globalOffset, Stagger(0b100), Stagger(0b000),
                            b3.p.pitch);

  // Do the actual computation here
  EdotB[globalOffset] +=
      delta * (E1 * B1 + E2 * B2 + E3 * B3) / sqrt(B1 * B1 + B2 * B2 + B3 * B3);
}

template <typename T>
__global__ void compute_EdotB_1d(pitchptr<T> e1, pitchptr<T> e2, pitchptr<T> e3,
                                 pitchptr<T> b1, pitchptr<T> b2, pitchptr<T> b3,
                                 pitchptr<T> EdotB) {
  // Compute time-averaged EdotB over the output interval
  int t1 = blockIdx.x;
  int c1 = threadIdx.x;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;

  size_t globalOffset = e1.compute_offset(n1);

  float delta = 1.0f / dev_params.data_interval;
  Scalar E1 = interpolate1d(e1, globalOffset, Stagger(0b110), Stagger(0b000));
  Scalar E2 = interpolate1d(e2, globalOffset, Stagger(0b101), Stagger(0b000));
  Scalar E3 = interpolate1d(e3, globalOffset, Stagger(0b011), Stagger(0b000));
  Scalar B1 = interpolate1d(b1, globalOffset, Stagger(0b001), Stagger(0b000));
  Scalar B2 = interpolate1d(b2, globalOffset, Stagger(0b010), Stagger(0b000));
  Scalar B3 = interpolate1d(b3, globalOffset, Stagger(0b100), Stagger(0b000));

  // Do the actual computation here
  EdotB[globalOffset] +=
      delta * (E1 * B1 + E2 * B2 + E3 * B3) / sqrt(B1 * B1 + B2 * B2 + B3 * B3);
}

__global__ void check_bg_fields() {
  printf("bg field has %lu, %lu, %lu\n", dev_bg_fields.B1.p.pitch,
         dev_bg_fields.B1.p.xsize, dev_bg_fields.B1.p.ysize);
  printf("bg field has %lu, %lu, %lu\n", dev_bg_fields.B2.p.pitch,
         dev_bg_fields.B2.p.xsize, dev_bg_fields.B2.p.ysize);
  // printf("bg B0 value is %f\n", *ptrAddr(dev_bg_fields.B1, 5, 4));
}

__global__ void check_dev_mesh() {
  printf("%d %d\n", dev_mesh.dims[0], dev_mesh.dims[1]);
  printf("%f %f\n", dev_mesh.lower[0], dev_mesh.lower[1]);
}

__global__ void fill_particles(particle_data ptc, size_t number, Scalar weight,
                               int multiplicity, curandState *states) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  CudaRng rng(&states[tid]);
  for (uint32_t cell = tid; cell < dev_mesh.size();
       cell += blockDim.x * gridDim.x) {
    if (!dev_mesh.is_in_bulk(cell))
      continue;
    // int Np = 3;
    for (int n = 0; n < multiplicity; n++) {
      size_t idx = number + cell * multiplicity * 2 + n * 2;
      float u = rng();
      ptc.x1[idx] = ptc.x1[idx + 1] = rng();
      ptc.x2[idx] = ptc.x2[idx + 1] = rng();
      ptc.x3[idx] = ptc.x3[idx + 1] = rng();
      ptc.p1[idx] = ptc.p1[idx + 1] = 0.0f;
      ptc.p2[idx] = ptc.p2[idx + 1] = 0.0f;
      ptc.p3[idx] = ptc.p3[idx + 1] = 0.0f;
      // ptc.E[idx] = ptc.E[idx + 1] = 1.0f;
      ptc.E[idx] = sqrt(1.0f + ptc.p1[idx] * ptc.p1[idx] +
                        ptc.p2[idx] * ptc.p2[idx] + ptc.p3[idx] * ptc.p3[idx]);
      ptc.E[idx + 1] = sqrt(1.0f + ptc.p1[idx + 1] * ptc.p1[idx + 1] +
                            ptc.p2[idx + 1] * ptc.p2[idx + 1] +
                            ptc.p3[idx + 1] * ptc.p3[idx + 1]);
      ptc.cell[idx] = ptc.cell[idx + 1] = cell;
      ptc.weight[idx] = ptc.weight[idx + 1] = weight;
      ptc.flag[idx] = set_ptc_type_flag(0, ParticleType::electron);
      ptc.flag[idx + 1] = set_ptc_type_flag(0, ParticleType::positron);
    }
  }
}

} // namespace Kernels
void sim_data::initialize(sim_environment &env) {
  init_bg_fields();

  g_ptrs.E1 = get_pitchptr(E.data(0));
  g_ptrs.E2 = get_pitchptr(E.data(1));
  g_ptrs.E3 = get_pitchptr(E.data(2));
  g_ptrs.B1 = get_pitchptr(B.data(0));
  g_ptrs.B2 = get_pitchptr(B.data(1));
  g_ptrs.B3 = get_pitchptr(B.data(2));
  g_ptrs.Ebg1 = get_pitchptr(Ebg.data(0));
  g_ptrs.Ebg2 = get_pitchptr(Ebg.data(1));
  g_ptrs.Ebg3 = get_pitchptr(Ebg.data(2));
  g_ptrs.Bbg1 = get_pitchptr(Bbg.data(0));
  g_ptrs.Bbg2 = get_pitchptr(Bbg.data(1));
  g_ptrs.Bbg3 = get_pitchptr(Bbg.data(2));
  g_ptrs.J1 = get_pitchptr(J.data(0));
  g_ptrs.J2 = get_pitchptr(J.data(1));
  g_ptrs.J3 = get_pitchptr(J.data(2));
  g_ptrs.divE = get_pitchptr(divE.data());
  g_ptrs.divB = get_pitchptr(divB.data());
  g_ptrs.EdotB = get_pitchptr(EdotB.data());
  g_ptrs.photon_produced = get_pitchptr(photon_produced.data());
  g_ptrs.pair_produced = get_pitchptr(pair_produced.data());
  g_ptrs.photon_num = get_pitchptr(photon_num.data());
  g_ptrs.ph_flux = get_pitchptr(ph_flux);

  CudaSafeCall(
      cudaMallocManaged(&g_ptrs.Rho, num_species * sizeof(pitchptr<Scalar>)));
  CudaSafeCall(
      cudaMallocManaged(&g_ptrs.gamma, num_species * sizeof(pitchptr<Scalar>)));
  CudaSafeCall(cudaMallocManaged(&g_ptrs.ptc_num,
                                 num_species * sizeof(pitchptr<Scalar>)));
  for (int n = 0; n < num_species; n++) {
    g_ptrs.Rho[n] = get_pitchptr(Rho[n].data());
    g_ptrs.gamma[n] = get_pitchptr(gamma[n].data());
    g_ptrs.ptc_num[n] = get_pitchptr(ptc_num[n].data());
  }

  visit_struct::for_each(g_ptrs.particles, particles.data(),
                         [](const char *name, auto &u, auto &v) { u = v; });
  visit_struct::for_each(g_ptrs.photons, photons.data(),
                         [](const char *name, auto &u, auto &v) { u = v; });

  int seed = env.params().random_seed;

  CudaSafeCall(cudaMalloc(&d_rand_states, 1024 * 512 * sizeof(curandState)));
  init_rand_states((curandState *)d_rand_states, seed, 1024, 512);
}

void sim_data::finalize() {
  CudaSafeCall(cudaFree(g_ptrs.Rho));
  CudaSafeCall(cudaFree(g_ptrs.gamma));
  CudaSafeCall(cudaFree(g_ptrs.ptc_num));
  cudaFree((curandState *)d_rand_states);
}

void sim_data::init_bg_fields() {
  init_dev_bg_fields(Ebg, Bbg);
  Kernels::check_bg_fields<<<1, 1>>>();
  CudaCheckError();
}

void sim_data::check_dev_mesh() {
  Kernels::check_dev_mesh<<<1, 1>>>();
  CudaCheckError();
}

void sim_data::compute_edotb() {
  auto &grid = env.grid();
  auto &mesh = grid.mesh();
  if (grid.dim() == 3) {
    dim3 blockSize(32, 8, 4);
    dim3 gridSize((mesh.reduced_dim(0) + blockSize.x - 1) / blockSize.x,
                  (mesh.reduced_dim(1) + blockSize.y - 1) / blockSize.y,
                  (mesh.reduced_dim(2) + blockSize.z - 1) / blockSize.z);
    Kernels::compute_EdotB_3d<<<gridSize, blockSize>>>(
        get_pitchptr(E, 0), get_pitchptr(E, 1), get_pitchptr(E, 2),
        get_pitchptr(B, 0), get_pitchptr(B, 1), get_pitchptr(B, 2),
        get_pitchptr(EdotB));
    CudaCheckError();
  } else if (grid.dim() == 2) {
    dim3 blockSize(32, 16);
    dim3 gridSize((mesh.reduced_dim(0) + blockSize.x - 1) / blockSize.x,
                  (mesh.reduced_dim(1) + blockSize.y - 1) / blockSize.y);
    Kernels::compute_EdotB_2d<<<gridSize, blockSize>>>(
        get_pitchptr(E, 0), get_pitchptr(E, 1), get_pitchptr(E, 2),
        get_pitchptr(B, 0), get_pitchptr(B, 1), get_pitchptr(B, 2),
        get_pitchptr(EdotB));
    CudaCheckError();
  } else if (grid.dim() == 1) { //
    dim3 blockSize(512);
    dim3 gridSize((mesh.reduced_dim(0) + blockSize.x - 1) / blockSize.x);
    Kernels::compute_EdotB_1d<<<gridSize, blockSize>>>(
        get_pitchptr(E, 0), get_pitchptr(E, 1), get_pitchptr(E, 2),
        get_pitchptr(B, 0), get_pitchptr(B, 1), get_pitchptr(B, 2),
        get_pitchptr(EdotB));
    CudaCheckError();
  }
}

void sim_data::fill_multiplicity(Scalar weight, int multiplicity) {
  int num_cells = env.mesh().size();
  int blockSize = 512;
  int gridSize = std::min((num_cells + blockSize - 1) / blockSize, 512);
  Kernels::fill_particles<<<gridSize, blockSize>>>(
      particles.data(), particles.number(), weight, multiplicity,
      (curandState *)d_rand_states);
  // cudaDeviceSynchronize();
  CudaCheckError();

  auto &mesh = env.mesh();
  particles.set_num(particles.number() +
                    mesh.size() * 2 * multiplicity);
  particles.sort_by_cell(env.grid());
  CudaSafeCall(cudaDeviceSynchronize());
}

data_ptrs get_data_ptrs(sim_data &data) { return g_ptrs; }

} // namespace Aperture
