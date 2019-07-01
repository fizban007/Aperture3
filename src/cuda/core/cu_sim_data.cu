#include "cu_sim_data.h"
#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cudaUtility.h"
#include "cuda/grids/grid_1dgr_dev.h"
#include "cuda/grids/grid_log_sph_dev.h"
#include "cuda/ptr_util.h"
#include "cuda/utils/iterate_devices.h"
#include "cuda/utils/pitchptr.cuh"
#include "utils/timer.h"
#include "visit_struct/visit_struct.hpp"

namespace Aperture {

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
check_mesh_ptrs(Grid_LogSph_dev::mesh_ptrs mesh_ptrs) {
  printf("mesh ptr %lu, %lu\n", mesh_ptrs.A1_e.p.pitch,
         mesh_ptrs.dV.p.pitch);
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

cu_sim_data::cu_sim_data(const cu_sim_environment &e)
    : env(e),
      particles(env.params().max_ptc_number),
      photons(env.params().max_photon_number) {
  num_species = env.params().num_species;
  Rho.resize(num_species);
  gamma.resize(num_species);
  ptc_num.resize(num_species);
  initialize(e);

  Logger::print_info("data_ptr has size", sizeof(data_ptrs));
}

cu_sim_data::~cu_sim_data() {
  if (ptrs.Rho != nullptr)
    CudaSafeCall(cudaFree(ptrs.Rho));
  if (ptrs.gamma != nullptr)
    CudaSafeCall(cudaFree(ptrs.gamma));
  if (ptrs.ptc_num != nullptr)
    CudaSafeCall(cudaFree(ptrs.ptc_num));
}

void
cu_sim_data::initialize(const cu_sim_environment &env) {
  // init_grid(env);

  // int dev_id = dev_map[n];
  CudaSafeCall(cudaSetDevice(devId));
  Logger::print_debug("using device {}", devId);
  auto &grid = env.local_grid();
  auto &mesh = grid.mesh();
  Logger::print_debug("Mesh dims are {}x{}x{}", mesh.dims[0],
                      mesh.dims[1], mesh.dims[2]);
  E = cu_vector_field<Scalar>(grid);
  E.set_field_type(FieldType::E);
  E.initialize();
  ptrs.E1 = E.ptr(0);
  ptrs.E2 = E.ptr(1);
  ptrs.E3 = E.ptr(2);

  Ebg = cu_vector_field<Scalar>(grid);
  Ebg.set_field_type(FieldType::E);
  Ebg.initialize();
  ptrs.Ebg1 = Ebg.ptr(0);
  ptrs.Ebg2 = Ebg.ptr(1);
  ptrs.Ebg3 = Ebg.ptr(2);

  B = cu_vector_field<Scalar>(grid);
  B.set_field_type(FieldType::B);
  B.initialize();
  ptrs.B1 = B.ptr(0);
  ptrs.B2 = B.ptr(1);
  ptrs.B3 = B.ptr(2);

  Bbg = cu_vector_field<Scalar>(grid);
  Bbg.set_field_type(FieldType::B);
  Bbg.initialize();
  ptrs.Bbg1 = Bbg.ptr(0);
  ptrs.Bbg2 = Bbg.ptr(1);
  ptrs.Bbg3 = Bbg.ptr(2);

  J = cu_vector_field<Scalar>(grid);
  J.set_field_type(FieldType::E);
  J.initialize();
  ptrs.J1 = J.ptr(0);
  ptrs.J2 = J.ptr(1);
  ptrs.J3 = J.ptr(2);

  flux = cu_scalar_field<Scalar>(grid);
  flux.initialize();

  divE = cu_scalar_field<Scalar>(grid);
  divB = cu_scalar_field<Scalar>(grid);
  divB.set_stagger(0b000);
  EdotB = cu_scalar_field<Scalar>(grid);
  EdotB.set_stagger(0b000);
  ptrs.divE = divE.ptr();
  ptrs.divB = divB.ptr();
  ptrs.EdotB = EdotB.ptr();

  photon_produced = cu_scalar_field<Scalar>(grid);
  photon_produced.initialize();
  pair_produced = cu_scalar_field<Scalar>(grid);
  pair_produced.initialize();
  photon_num = cu_scalar_field<Scalar>(grid);
  photon_num.initialize();
  ptrs.photon_produced = photon_produced.ptr();
  ptrs.pair_produced = pair_produced.ptr();
  ptrs.photon_num = photon_num.ptr();

  CudaSafeCall(cudaMallocManaged(&ptrs.Rho, num_species * sizeof(pitchptr<Scalar>)));
  CudaSafeCall(cudaMallocManaged(&ptrs.gamma, num_species * sizeof(pitchptr<Scalar>)));
  CudaSafeCall(cudaMallocManaged(&ptrs.ptc_num, num_species * sizeof(pitchptr<Scalar>)));
  for (int i = 0; i < num_species; i++) {
    Rho[i] = cu_scalar_field<Scalar>(grid);
    Rho[i].initialize();
    Rho[i].sync_to_host();
    ptrs.Rho[i] = Rho[i].ptr();

    gamma[i] = cu_scalar_field<Scalar>(grid);
    gamma[i].initialize();
    gamma[i].sync_to_host();
    ptrs.gamma[i] = gamma[i].ptr();

    ptc_num[i] = cu_scalar_field<Scalar>(grid);
    ptc_num[i].initialize();
    ptc_num[i].sync_to_host();
    ptrs.ptc_num[i] = ptc_num[i].ptr();
  }

  CudaSafeCall(cudaDeviceSynchronize());

  init_bg_fields();
}

void
cu_sim_data::fill_multiplicity(Scalar weight, int multiplicity) {
  Kernels::fill_particles<<<dim3(16, 16), dim3(32, 32)>>>(
      particles.data(), particles.number(), weight, multiplicity);
  // cudaDeviceSynchronize();
  CudaCheckError();

  auto &mesh = env.local_grid().mesh();
  particles.set_num(particles.number() +
                    mesh.dims[0] * mesh.dims[1] * 2 * multiplicity);
  CudaSafeCall(cudaDeviceSynchronize());
}

// void
// cu_sim_data::init_grid(const cu_sim_environment &env) {
  // grid.resize(dev_map.size());
  // int last_dim = env.grid().dim() - 1;
  // int offset = 0;
  // for (int n = 0; n < dev_map.size(); n++) {
  //   int dev_id = dev_map[n];
  //   CudaSafeCall(cudaSetDevice(dev_id));

  //   // Setup the grids
  //   if (env.params().coord_system == "Cartesian") {
  //     grid[n].reset(new Grid());
  //   } else if (env.params().coord_system == "LogSpherical") {
  //     grid[n].reset(new Grid_LogSph_dev());
  //   } else if (env.params().coord_system == "1DGR" &&
  //              grid[n]->dim() == 1) {
  //     grid[n].reset(new Grid_1dGR_dev());
  //   } else {
  //     grid[n].reset(new Grid());
  //   }
  //   grid[n]->init(env.sub_params(n));
  //   auto &mesh = grid[n]->mesh();
  //   mesh.offset[last_dim] = offset;
  //   // if (offset == 0) offset += mesh.guard[last_dim];
  //   offset += mesh.reduced_dim(last_dim);
  //   Logger::print_debug("Grid dimension for dev {} is {}x{}x{}", dev_id,
  //                       mesh.dims[0], mesh.dims[1], mesh.dims[2]);
  //   Logger::print_debug("Grid lower are {}, {}, {}", mesh.lower[0],
  //                       mesh.lower[1], mesh.lower[2]);
  //   if (grid[n]->mesh().delta[0] < env.params().delta_t) {
  //     std::cerr
  //         << "Grid spacing should be larger than delta_t! Aborting!"
  //         << std::endl;
  //     abort();
  //   }
  //   init_dev_mesh(grid[n]->mesh());
  // }
  // check_dev_mesh();
  // check_mesh_ptrs();
// }

void
cu_sim_data::send_particles() {
  timer::stamp("send_ptc");
  std::vector<int *> ptc_send_left(dev_map.size());
  std::vector<int *> ptc_send_right(dev_map.size());
  std::vector<int *> ph_send_left(dev_map.size());
  std::vector<int *> ph_send_right(dev_map.size());
  for (int i = 0; i < dev_map.size(); i++) {
    CudaSafeCall(cudaSetDevice(dev_map[i]));
    CudaSafeCall(cudaMallocManaged(&(ptc_send_left[i]), sizeof(int)));
    *ptc_send_left[i] = 0;
    CudaSafeCall(cudaMallocManaged(&(ptc_send_right[i]), sizeof(int)));
    *ptc_send_right[i] = 0;
    CudaSafeCall(cudaMallocManaged(&(ph_send_left[i]), sizeof(int)));
    *ph_send_left[i] = 0;
    CudaSafeCall(cudaMallocManaged(&(ph_send_right[i]), sizeof(int)));
    *ph_send_right[i] = 0;
  }
  int last_dim = grid[0]->dim() - 1;
  // Logger::print_debug("last_dim is {}", last_dim);
  for (int n = 0; n < dev_map.size(); n++) {
    CudaSafeCall(cudaSetDevice(dev_map[n]));
    Kernels::send_particles<<<256, 512>>>(
        particles[n].data(), particles[n].number(),
        ptc_buffer[2 * n].data(), ptc_buffer[2 * n + 1].data(),
        last_dim, ptc_send_left[n], ptc_send_right[n]);
    CudaCheckError();

    Kernels::send_photons<<<256, 512>>>(
        photons[n].data(), photons[n].number(), ph_buffer[2 * n].data(),
        ph_buffer[2 * n + 1].data(), last_dim, ph_send_left[n],
        ph_send_right[n]);
    CudaCheckError();
  }
  for_each_device(dev_map,
                  [](int n) { CudaSafeCall(cudaDeviceSynchronize()); });
  for (int n = 1; n < dev_map.size(); n++) {
    CudaSafeCall(cudaSetDevice(dev_map[n - 1]));
    visit_struct::for_each(
        particles[n - 1].data(), ptc_buffer[2 * n].data(),
        [&](const char *name, auto &x1, auto &x2) {
          typedef typename std::remove_reference<decltype(*x1)>::type
              x_type;
          CudaSafeCall(cudaMemcpyPeer(
              x1 + particles[n - 1].number(), dev_map[n - 1], x2,
              dev_map[n], *ptc_send_left[n] * sizeof(x_type)));
        });
    particles[n - 1].set_num(particles[n - 1].number() +
                             *ptc_send_left[n]);
    Logger::print_info("Sending {} particles from dev {} to dev {}",
                       *ptc_send_left[n], n, n - 1);
  }
  for (int n = 0; n < dev_map.size() - 1; n++) {
    CudaSafeCall(cudaSetDevice(dev_map[n + 1]));
    visit_struct::for_each(
        particles[n + 1].data(), ptc_buffer[2 * n + 1].data(),
        [&](const char *name, auto &x1, auto &x2) {
          typedef typename std::remove_reference<decltype(*x1)>::type
              x_type;
          CudaSafeCall(cudaMemcpyPeer(
              x1 + particles[n + 1].number(), dev_map[n + 1], x2,
              dev_map[n], *ptc_send_right[n] * sizeof(x_type)));
        });
    particles[n + 1].set_num(particles[n + 1].number() +
                             *ptc_send_right[n]);
    Logger::print_info("Sending {} particles from dev {} to dev {}",
                       *ptc_send_right[n], n, n + 1);
  }
  for (int n = 1; n < dev_map.size(); n++) {
    CudaSafeCall(cudaSetDevice(dev_map[n - 1]));
    visit_struct::for_each(
        photons[n - 1].data(), ph_buffer[2 * n].data(),
        [&](const char *name, auto &x1, auto &x2) {
          typedef typename std::remove_reference<decltype(*x1)>::type
              x_type;
          CudaSafeCall(cudaMemcpyPeer(
              x1 + photons[n - 1].number(), dev_map[n - 1], x2,
              dev_map[n], *ph_send_left[n] * sizeof(x_type)));
        });
    photons[n - 1].set_num(photons[n - 1].number() + *ph_send_left[n]);
  }
  for (int n = 0; n < dev_map.size() - 1; n++) {
    CudaSafeCall(cudaSetDevice(dev_map[n + 1]));
    visit_struct::for_each(
        photons[n + 1].data(), ph_buffer[2 * n + 1].data(),
        [&](const char *name, auto &x1, auto &x2) {
          typedef typename std::remove_reference<decltype(*x1)>::type
              x_type;
          CudaSafeCall(cudaMemcpyPeer(
              x1 + photons[n + 1].number(), dev_map[n + 1], x2,
              dev_map[n], *ph_send_right[n] * sizeof(x_type)));
        });
    photons[n + 1].set_num(photons[n + 1].number() + *ph_send_right[n]);
  }
  for_each_device(dev_map,
                  [](int n) { CudaSafeCall(cudaDeviceSynchronize()); });
  // for (int i = 0; i < dev_map.size(); i++) {
  //   if (particles[i].number() == 0)
  //     continue;
  //   CudaSafeCall(cudaSetDevice(dev_map[i]));
  //   if (i > 0) {
  //     // Send particles right
  //     Kernels::send_particles<<<256, 512>>>(
  //         particles[i - 1].data(), particles[i - 1].number(),
  //         particles[i].data(), particles[i].number(),
  //         ptc_recv_left[i], grid[0]->dim() - 1, 1);
  //     CudaCheckError();
  //   }
  //   if (i < dev_map.size() - 1) {
  //     // Send particles left
  //     Kernels::send_particles<<<256, 512>>>(
  //         particles[i + 1].data(), particles[i + 1].number(),
  //         particles[i].data(), particles[i].number(),
  //         ptc_recv_right[i], grid[0]->dim() - 1, 0);
  //     CudaCheckError();
  //   }
  // }
  // for (int i = 0; i < dev_map.size(); i++) {
  //   if (photons[i].number() == 0)
  //     continue;
  //   CudaSafeCall(cudaSetDevice(dev_map[i]));
  //   if (i > 0) {
  //     // Send particles right
  //     Kernels::send_photons<<<256, 512>>>(
  //         photons[i - 1].data(), photons[i - 1].number(),
  //         photons[i].data(), photons[i].number(), ph_recv_left[i],
  //         grid[0]->dim() - 1, 1);
  //     CudaCheckError();
  //   }
  //   if (i < dev_map.size() - 1) {
  //     // Send particles left
  //     Kernels::send_photons<<<256, 512>>>(
  //         photons[i + 1].data(), photons[i + 1].number(),
  //         photons[i].data(), photons[i].number(), ph_recv_right[i],
  //         grid[0]->dim() - 1, 0);
  //     CudaCheckError();
  //   }
  // }
  // for (int i = 0; i < dev_map.size(); i++) {
  //   CudaSafeCall(cudaSetDevice(dev_map[i]));
  //   CudaSafeCall(cudaDeviceSynchronize());
  //   if (i > 0) {
  //     particles[i].set_num(particles[i].number() +
  //     *ptc_recv_left[i]); photons[i].set_num(photons[i].number() +
  //     *ph_recv_left[i]); Logger::print_info("Sent {} particles from
  //     dev {} to dev {}", *ptc_recv_left[i],
  //                        i - 1, i);
  //   }
  //   if (i < dev_map.size() - 1) {
  //     particles[i].set_num(particles[i].number() +
  //     *ptc_recv_right[i]); photons[i].set_num(photons[i].number() +
  //                            *ph_recv_right[i]);
  //     Logger::print_info("Sent {} particles from dev {} to dev {}",
  //     *ptc_recv_right[i],
  //                        i + 1, i);
  //   }
  // }
  for_each_device(dev_map, [&](int i) {
    CudaSafeCall(cudaFree(ptc_send_left[i]));
    CudaSafeCall(cudaFree(ptc_send_right[i]));
    CudaSafeCall(cudaFree(ph_send_left[i]));
    CudaSafeCall(cudaFree(ph_send_right[i]));
  });
  timer::show_duration_since_stamp("Sending particles", "ms",
                                   "send_ptc");
}

void
cu_sim_data::sort_particles() {
  timer::stamp("ptc_sort");
  particles.sort_by_cell();
  photons.sort_by_cell();
  timer::show_duration_since_stamp("Sorting particles", "us",
                                   "ptc_sort");
}

void
cu_sim_data::init_bg_fields() {
  Logger::print_debug("on host, B0 is {}", Bbg(0, 5, 4));
  init_dev_bg_fields(Ebg, Bbg);
  Kernels::check_bg_fields<<<1, 1>>>();
  CudaCheckError();
}

void
cu_sim_data::compute_edotb() {
  auto &mesh = env.local_grid().mesh();
  dim3 blockSize(32, 16);
  dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 16);
  Kernels::compute_EdotB<<<gridSize, blockSize>>>(
      E.ptr(0), E.ptr(1), E.ptr(2), B.ptr(0), B.ptr(1),
      B.ptr(2), Bbg.ptr(0), Bbg.ptr(1), Bbg.ptr(2),
      EdotB.ptr());
  CudaCheckError();
}

void
cu_sim_data::check_dev_mesh() {
  Kernels::check_dev_mesh<<<1, 1>>>();
  CudaCheckError();
}

void
cu_sim_data::check_mesh_ptrs() {
  const Grid_LogSph_dev &g =
      *dynamic_cast<const Grid_LogSph_dev *>(&env.local_grid());
  auto mesh_ptrs = g.get_mesh_ptrs();
  Kernels::check_mesh_ptrs<<<1, 1>>>(mesh_ptrs);
  CudaCheckError();
}

cu_sim_data::data_ptrs
cu_sim_data::get_ptrs() {
  return ptrs;
}

}  // namespace Aperture
