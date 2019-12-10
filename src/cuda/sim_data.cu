#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cudaUtility.h"
#include "cuda/data_ptrs.h"
#include "cuda/kernels.h"
#include "cuda/utils/pitchptr.h"
#include "sim_data_impl.hpp"
#include "visit_struct/visit_struct.hpp"

namespace Aperture {

static data_ptrs g_ptrs;

namespace Kernels {

template <typename T>
__global__ void
compute_EdotB(pitchptr<T> e1, pitchptr<T> e2, pitchptr<T> e3,
              pitchptr<T> b1, pitchptr<T> b2, pitchptr<T> b3,
              pitchptr<T> b1bg, pitchptr<T> b2bg, pitchptr<T> b3bg,
              pitchptr<T> EdotB) {
  // Compute time-averaged EdotB over the output interval
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  // size_t globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);
  size_t globalOffset = e1.compute_offset(n1, n2);

  float delta = 1.0f / dev_params.data_interval;
  Scalar E1 = 0.5f * (e1(n1, n2) + e1(n1, n2 - 1));
  Scalar E2 = 0.5f * (e2(n1, n2) + e2(n1 - 1, n2));
  Scalar E3 = 0.25f * (e3(n1, n2) + e3(n1 - 1, n2) + e3(n1, n2 - 1) +
                       e3(n1 - 1, n2 - 1));
  Scalar B1 = 0.5f * (b1(n1, n2) + b1(n1 - 1, n2)) +
              0.5f * (b1bg(n1, n2) + b1bg(n1 - 1, n2));
  Scalar B2 = 0.5f * (b2(n1, n2) + b2(n1, n2 - 1)) +
              0.5f * (b2bg(n1, n2) + b2bg(n1, n2 - 1));
  Scalar B3 = b3[globalOffset] + b3bg[globalOffset];

  // Do the actual computation here
  EdotB[globalOffset] += delta * (E1 * B1 + E2 * B2 + E3 * B3) /
                         sqrt(B1 * B1 + B2 * B2 + B3 * B3);
}

__global__ void
check_bg_fields() {
  printf("bg field has %lu, %lu, %lu\n", dev_bg_fields.B1.p.pitch,
         dev_bg_fields.B1.p.xsize, dev_bg_fields.B1.p.ysize);
  printf("bg field has %lu, %lu, %lu\n", dev_bg_fields.B2.p.pitch,
         dev_bg_fields.B2.p.xsize, dev_bg_fields.B2.p.ysize);
  // printf("bg B0 value is %f\n", *ptrAddr(dev_bg_fields.B1, 5, 4));
}

__global__ void
check_dev_mesh() {
  printf("%d %d\n", dev_mesh.dims[0], dev_mesh.dims[1]);
  printf("%f %f\n", dev_mesh.lower[0], dev_mesh.lower[1]);
}

__global__ void
fill_particles(particle_data ptc, size_t number, Scalar weight,
               int multiplicity) {
  for (int j =
           blockIdx.y * blockDim.y + threadIdx.y + dev_mesh.guard[1];
       j < dev_mesh.dims[1] - dev_mesh.guard[1];
       j += blockDim.y * gridDim.y) {
    for (int i =
             blockIdx.x * blockDim.x + threadIdx.x + dev_mesh.guard[0];
         i < dev_mesh.dims[0] - dev_mesh.guard[0];
         i += blockDim.x * gridDim.x) {
      uint32_t cell = i + j * dev_mesh.dims[0];
      Scalar theta = dev_mesh.pos(1, j, 0.5f);
      // int Np = 3;
      for (int n = 0; n < multiplicity; n++) {
        size_t idx = number + cell * multiplicity * 2 + n * 2;
        ptc.x1[idx] = ptc.x1[idx + 1] = 0.5f;
        ptc.x2[idx] = ptc.x2[idx + 1] = 0.5f;
        ptc.x3[idx] = ptc.x3[idx + 1] = 0.0f;
        ptc.p1[idx] = ptc.p1[idx + 1] = 0.0f;
        ptc.p2[idx] = ptc.p2[idx + 1] = 0.0f;
        ptc.p3[idx] = ptc.p3[idx + 1] = 0.0f;
        ptc.E[idx] = ptc.E[idx + 1] = 1.0f;
        ptc.cell[idx] = ptc.cell[idx + 1] = cell;
        ptc.weight[idx] = ptc.weight[idx + 1] = weight * sin(theta);
        ptc.flag[idx] = set_ptc_type_flag(0, ParticleType::electron);
        ptc.flag[idx + 1] =
            set_ptc_type_flag(0, ParticleType::positron);
      }
    }
  }
}

// __global__ void
// send_particles(particle_data ptc_src, size_t num_src,
//                particle_data ptc_dst, size_t num_dst, int *ptc_sent,
//                int dim, int dir) {
//   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_src;
//        i += blockDim.x * gridDim.x) {
//     uint32_t cell = ptc_src.cell[i];
//     int c, delta_cell;
//     if (dim == 0) {
//       c = dev_mesh.get_c1(cell);
//       delta_cell = dev_mesh.reduced_dim(0);
//     } else if (dim == 1) {
//       c = dev_mesh.get_c2(cell);
//       delta_cell = dev_mesh.reduced_dim(1) * dev_mesh.dims[0];
//     } else if (dim == 2) {
//       c = dev_mesh.get_c3(cell);
//       delta_cell =
//           dev_mesh.reduced_dim(2) * dev_mesh.dims[0] *
//           dev_mesh.dims[1];
//     }
//     if ((dir == 0 && c < dev_mesh.guard[dim]) ||
//         (dir == 1 && c >= dev_mesh.dims[dim] - dev_mesh.guard[dim]))
//         {
//       size_t pos = atomicAdd(ptc_sent, 1) + num_dst;
//       ptc_src.cell[i] = MAX_CELL;
//       ptc_dst.cell[pos] = cell + (dir == 0 ? 1 : -1) * delta_cell;
//       ptc_dst.x1[pos] = ptc_src.x1[i];
//       ptc_dst.x2[pos] = ptc_src.x2[i];
//       ptc_dst.x3[pos] = ptc_src.x3[i];
//       ptc_dst.p1[pos] = ptc_src.p1[i];
//       ptc_dst.p2[pos] = ptc_src.p2[i];
//       ptc_dst.p3[pos] = ptc_src.p3[i];
//       ptc_dst.E[pos] = ptc_src.E[i];
//       ptc_dst.weight[pos] = ptc_src.weight[i];
//       ptc_dst.flag[pos] = ptc_src.flag[i];
//     }
//   }
// }

// __global__ void
// send_particles(particle_data ptc_src, size_t num_src,
//                particle_data buffer_left, particle_data buffer_right,
//                int dim, int *ptc_sent_left, int *ptc_sent_right) {
//   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_src;
//        i += blockDim.x * gridDim.x) {
//     uint32_t cell = ptc_src.cell[i];
//     int c, delta_cell;
//     if (dim == 0) {
//       c = dev_mesh.get_c1(cell);
//       delta_cell = dev_mesh.reduced_dim(0);
//     } else if (dim == 1) {
//       c = dev_mesh.get_c2(cell);
//       delta_cell = dev_mesh.reduced_dim(1) * dev_mesh.dims[0];
//     } else if (dim == 2) {
//       c = dev_mesh.get_c3(cell);
//       delta_cell =
//           dev_mesh.reduced_dim(2) * dev_mesh.dims[0] *
//           dev_mesh.dims[1];
//     }
//     if (c < dev_mesh.guard[dim]) {
//       size_t pos = atomicAdd(ptc_sent_left, 1);
//       ptc_src.cell[i] = MAX_CELL;
//       buffer_left.cell[pos] = cell + delta_cell;
//       buffer_left.x1[pos] = ptc_src.x1[i];
//       buffer_left.x2[pos] = ptc_src.x2[i];
//       buffer_left.x3[pos] = ptc_src.x3[i];
//       buffer_left.p1[pos] = ptc_src.p1[i];
//       buffer_left.p2[pos] = ptc_src.p2[i];
//       buffer_left.p3[pos] = ptc_src.p3[i];
//       buffer_left.E[pos] = ptc_src.E[i];
//       buffer_left.weight[pos] = ptc_src.weight[i];
//       buffer_left.flag[pos] = ptc_src.flag[i];
//     } else if (c >= dev_mesh.dims[dim] - dev_mesh.guard[dim]) {
//       size_t pos = atomicAdd(ptc_sent_right, 1);
//       ptc_src.cell[i] = MAX_CELL;
//       buffer_right.cell[pos] = cell - delta_cell;
//       buffer_right.x1[pos] = ptc_src.x1[i];
//       buffer_right.x2[pos] = ptc_src.x2[i];
//       buffer_right.x3[pos] = ptc_src.x3[i];
//       buffer_right.p1[pos] = ptc_src.p1[i];
//       buffer_right.p2[pos] = ptc_src.p2[i];
//       buffer_right.p3[pos] = ptc_src.p3[i];
//       buffer_right.E[pos] = ptc_src.E[i];
//       buffer_right.weight[pos] = ptc_src.weight[i];
//       buffer_right.flag[pos] = ptc_src.flag[i];
//     }
//   }
// }

// __global__ void
// send_photons(photon_data ptc_src, size_t num_src, photon_data
// ptc_dst,
//              size_t num_dst, int *ptc_sent, int dim, int dir) {
//   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_src;
//        i += blockDim.x * gridDim.x) {
//     uint32_t cell = ptc_src.cell[i];
//     int c;
//     if (dim == 0) {
//       c = dev_mesh.get_c1(cell);
//     } else if (dim == 1) {
//       c = dev_mesh.get_c2(cell);
//     } else if (dim == 2) {
//       c = dev_mesh.get_c3(cell);
//     }
//     if ((dir == 0 && c < dev_mesh.guard[dim]) ||
//         (dir == 1 && c >= dev_mesh.dims[dim] - dev_mesh.guard[dim]))
//         {
//       size_t pos = atomicAdd(ptc_sent, 1) + num_dst;
//       ptc_src.cell[i] = MAX_CELL;
//       ptc_dst.cell[pos] = cell;
//       ptc_dst.x1[pos] = ptc_src.x1[i];
//       ptc_dst.x2[pos] = ptc_src.x2[i];
//       ptc_dst.x3[pos] = ptc_src.x3[i];
//       ptc_dst.p1[pos] = ptc_src.p1[i];
//       ptc_dst.p2[pos] = ptc_src.p2[i];
//       ptc_dst.p3[pos] = ptc_src.p3[i];
//       ptc_dst.E[pos] = ptc_src.E[i];
//       ptc_dst.weight[pos] = ptc_src.weight[i];
//       ptc_dst.path_left[pos] = ptc_src.path_left[i];
//       ptc_dst.flag[pos] = ptc_src.flag[i];
//     }
//   }
// }

// __global__ void
// send_photons(photon_data ph_src, size_t num_src,
//              photon_data buffer_left, photon_data buffer_right, int
//              dim, int *ph_sent_left, int *ph_sent_right) {
//   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_src;
//        i += blockDim.x * gridDim.x) {
//     uint32_t cell = ph_src.cell[i];
//     int c, delta_cell;
//     if (dim == 0) {
//       c = dev_mesh.get_c1(cell);
//       delta_cell = dev_mesh.reduced_dim(0);
//     } else if (dim == 1) {
//       c = dev_mesh.get_c2(cell);
//       delta_cell = dev_mesh.reduced_dim(1) * dev_mesh.dims[0];
//     } else if (dim == 2) {
//       c = dev_mesh.get_c3(cell);
//       delta_cell =
//           dev_mesh.reduced_dim(2) * dev_mesh.dims[0] *
//           dev_mesh.dims[1];
//     }
//     if (c < dev_mesh.guard[dim]) {
//       size_t pos = atomicAdd(ph_sent_left, 1);
//       ph_src.cell[i] = MAX_CELL;
//       buffer_left.cell[pos] = cell + delta_cell;
//       buffer_left.x1[pos] = ph_src.x1[i];
//       buffer_left.x2[pos] = ph_src.x2[i];
//       buffer_left.x3[pos] = ph_src.x3[i];
//       buffer_left.p1[pos] = ph_src.p1[i];
//       buffer_left.p2[pos] = ph_src.p2[i];
//       buffer_left.p3[pos] = ph_src.p3[i];
//       buffer_left.E[pos] = ph_src.E[i];
//       buffer_left.weight[pos] = ph_src.weight[i];
//       buffer_left.path_left[pos] = ph_src.path_left[i];
//       buffer_left.flag[pos] = ph_src.flag[i];
//     } else if (c >= dev_mesh.dims[dim] - dev_mesh.guard[dim]) {
//       size_t pos = atomicAdd(ph_sent_right, 1);
//       ph_src.cell[i] = MAX_CELL;
//       buffer_right.cell[pos] = cell - delta_cell;
//       buffer_right.x1[pos] = ph_src.x1[i];
//       buffer_right.x2[pos] = ph_src.x2[i];
//       buffer_right.x3[pos] = ph_src.x3[i];
//       buffer_right.p1[pos] = ph_src.p1[i];
//       buffer_right.p2[pos] = ph_src.p2[i];
//       buffer_right.p3[pos] = ph_src.p3[i];
//       buffer_right.E[pos] = ph_src.E[i];
//       buffer_right.weight[pos] = ph_src.weight[i];
//       buffer_right.path_left[pos] = ph_src.path_left[i];
//       buffer_right.flag[pos] = ph_src.flag[i];
//     }
//   }
// }

}  // namespace Kernels
void
sim_data::initialize(const sim_environment& env) {
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

  CudaSafeCall(cudaMallocManaged(
      &g_ptrs.Rho, num_species * sizeof(pitchptr<Scalar>)));
  CudaSafeCall(cudaMallocManaged(
      &g_ptrs.gamma, num_species * sizeof(pitchptr<Scalar>)));
  CudaSafeCall(cudaMallocManaged(
      &g_ptrs.ptc_num, num_species * sizeof(pitchptr<Scalar>)));
  for (int n = 0; n < num_species; n++) {
    g_ptrs.Rho[n] = get_pitchptr(Rho[n].data());
    g_ptrs.gamma[n] = get_pitchptr(gamma[n].data());
    g_ptrs.ptc_num[n] = get_pitchptr(ptc_num[n].data());
  }

  visit_struct::for_each(
      g_ptrs.particles, particles.data(),
      [](const char* name, auto& u, auto& v) { u = v; });
  visit_struct::for_each(
      g_ptrs.photons, photons.data(),
      [](const char* name, auto& u, auto& v) { u = v; });

  int seed = env.params().random_seed;

  CudaSafeCall(
      cudaMalloc(&d_rand_states, 1024 * 512 * sizeof(curandState)));
  init_rand_states((curandState*)d_rand_states, seed, 1024, 512);
}

void
sim_data::finalize() {
  CudaSafeCall(cudaFree(g_ptrs.Rho));
  CudaSafeCall(cudaFree(g_ptrs.gamma));
  CudaSafeCall(cudaFree(g_ptrs.ptc_num));
  cudaFree((curandState*)d_rand_states);
}

void
sim_data::init_bg_fields() {
  init_dev_bg_fields(Ebg, Bbg);
  Kernels::check_bg_fields<<<1, 1>>>();
  CudaCheckError();
}

void
sim_data::check_dev_mesh() {
  Kernels::check_dev_mesh<<<1, 1>>>();
  CudaCheckError();
}

void
sim_data::compute_edotb() {}

void
sim_data::fill_multiplicity(Scalar weight, int multiplicity) {
  Kernels::fill_particles<<<dim3(16, 16), dim3(32, 32)>>>(
      particles.data(), particles.number(), weight, multiplicity);
  // cudaDeviceSynchronize();
  CudaCheckError();

  auto& mesh = env.local_grid().mesh();
  particles.set_num(particles.number() +
                    mesh.dims[0] * mesh.dims[1] * 2 * multiplicity);
  CudaSafeCall(cudaDeviceSynchronize());
}

data_ptrs
get_data_ptrs(sim_data& data) {
  return g_ptrs;
}

}  // namespace Aperture
