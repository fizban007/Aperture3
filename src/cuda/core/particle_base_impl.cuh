#ifndef _PARTICLE_BASE_IMPL_CUH_
#define _PARTICLE_BASE_IMPL_CUH_

#include "cuda/constant_mem.h"
#include "cuda/cudaUtility.h"
// #include "cuda/data/particle_base_dev.h"
#include "core/particle_base.h"
#include "cuda/cuda_control.h"
#include "cuda/kernels.h"
#include "cuda/memory.h"
#include "utils/for_each_arg.hpp"
#include "utils/logger.h"
#include "utils/timer.h"

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/replace.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "visit_struct/visit_struct.hpp"
#include <algorithm>
#include <cstring>
#include <mpi.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
// #include "types/particles_dev.h"

namespace Aperture {

namespace Kernels {

// __global__ void
// compute_target_buffers(const uint32_t* cells, size_t num,
//                        int* buffer_num, size_t num_zones, size_t*
//                        idx) {
//   for (size_t n = threadIdx.x + blockIdx.x * blockDim.x; n < num;
//        n += blockDim.x * gridDim.x) {
//     int zone = dev_mesh.find_zone(cells[n]);
//     if (zone == num_zones / 2) continue;
//     if (zone >= num_zones / 2) zone -= 1;

//     int pos = atomicAdd(&buffer_num[zone], 1);
//     // Zone is less than 32, so we can use 5 bits to represent this
//     // hopefully pos won't be exceeding 1<<26 which is 6.7e7
//     idx[n] = ((pos & 0b11111) << 27) + pos;
//   }
// }

// template <typename T>
// __global__ void
// copy_component_to_buffer(T* ptr, size_t num, size_t* idx, ) {
//   for (size_t n = threadIdx.x + blockIdx.x * blockDim.x; n < num;
//        n += blockDim.x * gridDim.x) {
//     size_t i = idx[n];
//     int zone = ((i >> 27) & 0b11111);
//     int pos = i - (zone << 27);
//   }
// }

template <typename T>
__global__ void
get_tracked_ptc_attr(T* ptr, uint32_t* tracked_map, size_t num,
                     T* tracked_ptr) {
  for (size_t n = threadIdx.x + blockIdx.x * blockDim.x; n < num;
       n += blockDim.x * gridDim.x) {
    tracked_ptr[n] = ptr[tracked_map[n]];
  }
}

}  // namespace Kernels

// These are helper functors for for_each loops

struct assign_at_idx {
  size_t idx_;
  HOST_DEVICE assign_at_idx(size_t idx) : idx_(idx) {}

  template <typename T, typename U>
  HD_INLINE void operator()(T& t, const U& u) const {
    t[idx_] = u;
  }
};

struct fill_pos_amount {
  size_t pos_, amount_;

  fill_pos_amount(size_t pos, size_t amount)
      : pos_(pos), amount_(amount) {}

  template <typename T, typename U>
  void operator()(T& t, U& u) const {
    // std::fill_n(t + pos_, amount_, u);
    auto t_ptr = thrust::device_pointer_cast(t);
    thrust::fill_n(t_ptr + pos_, amount_, u);
    CudaCheckError();
  }
};

// struct sync_dev {
//   int devId_;
//   size_t size_;

//   sync_dev(int devId, size_t size) : devId_(devId), size_(size) {}

//   template <typename T>
//   void operator()(const char* name, T& x) const {
//     typedef typename std::remove_reference<decltype(*x)>::type
//     x_type; cudaMemPrefetchAsync(x, size_ * sizeof(x_type), devId_);
//   }
// };

struct copy_to_dest {
  size_t num_, src_pos_, dest_pos_;
  copy_to_dest(size_t num, size_t src_pos, size_t dest_pos)
      : num_(num), src_pos_(src_pos), dest_pos_(dest_pos) {}

  template <typename T, typename U>
  void operator()(T& t, U& u) const {
    auto t_ptr = thrust::device_pointer_cast(t);
    auto u_ptr = thrust::device_pointer_cast(u);
    thrust::copy_n(u_ptr + src_pos_, num_, t_ptr + dest_pos_);
    // std::copy(u + src_pos_, u + src_pos_ + num_, t + dest_pos_);
  }
};

// template <typename ArrayType>
struct rearrange_array {
  // ArrayType& array_;
  // thrust::device_ptr<Index_t>& index_;
  Index_t* index_;
  size_t N_;
  void* tmp_ptr_;
  std::string skip_;

  rearrange_array(Index_t* index, size_t N, void* tmp_ptr,
                  const std::string& skip)
      : index_(index), N_(N), tmp_ptr_(tmp_ptr), skip_(skip) {}

  template <typename T, typename U>
  void operator()(const char* name, T& x, U& u) const {
    auto ptr_index = thrust::device_pointer_cast(index_);
    // Logger::print_info("rearranging {}", name);
    if (std::strcmp(name, skip_.c_str()) == 0) {
      // Logger::print_info("skipping {}", name);
      return;
    }
    auto x_ptr = thrust::device_pointer_cast(x);
    auto tmp_ptr =
        thrust::device_pointer_cast(reinterpret_cast<U*>(tmp_ptr_));
    thrust::gather(ptr_index, ptr_index + N_, x_ptr, tmp_ptr);
    thrust::copy_n(tmp_ptr, N_, x_ptr);
    CudaCheckError();
  }
};

// struct compare_zone {
//   int zone;

//   HOST_DEVICE compare_zone(int z) : zone(z) {}

//   __device__ bool operator()(uint32_t c) {
//     return (dev_mesh.find_zone(c) == zone);
//   }
// };

// struct not_center_zone {
//   __device__ bool operator()(uint32_t c) {
//     return (dev_mesh.find_zone(c) != 13);
//   }
// };

struct modify_cell {
  int _dx, _dy, _dz;

  HOST_DEVICE modify_cell(int dx, int dy, int dz) :
      _dx(dx), _dy(dy), _dz(dz) {}

  __device__ uint32_t operator()(uint32_t c) {
    return c -
           _dz * dev_mesh.reduced_dim(2) * dev_mesh.dims[0] *
               dev_mesh.dims[1] -
           _dy * dev_mesh.reduced_dim(1) * dev_mesh.dims[0] -
           _dx * dev_mesh.reduced_dim(0);
  }
};

template <typename ParticleClass>
particle_base<ParticleClass>::particle_base()
    : m_tmp_data_ptr(nullptr), m_index(nullptr) {
  visit_struct::for_each(
      m_data, [](const char* name, auto& x) { x = nullptr; });
}

template <typename ParticleClass>
particle_base<ParticleClass>::particle_base(std::size_t max_num,
                                            bool managed) {
  m_size = max_num;
  m_managed = managed;
  // std::cout << "New particle array with size " << max_num <<
  // std::endl;
  alloc_mem(max_num, managed);
  // auto alloc = alloc_cuda_managed(max_num);
  // auto alloc = alloc_cuda_device(max_num);
  // alloc("index", m_index);
  cudaMalloc(&m_index, max_num * sizeof(Index_t));
  cudaMalloc(&m_tmp_data_ptr, max_num * sizeof(double));
  // m_index.resize(max_num, 0);
  // m_index_bak.resize(max_num, 0);
  initialize();
}

// template <typename ParticleClass>
// particle_base<ParticleClass>::particle_base(
//     const particle_base<ParticleClass>& other) {
//   m_size = other.m_size;
//   m_number = other.m_number;
//   m_managed = other.m_managed;

//   alloc_mem(m_size, m_managed);
//   // auto alloc = alloc_cuda_managed(n);
//   // auto alloc = alloc_cuda_device(n);
//   // alloc(m_index);
//   cudaMalloc(&m_index, m_size * sizeof(Index_t));
//   cudaMalloc(&m_tmp_data_ptr, m_size * sizeof(double));
//   // alloc((double*)m_tmp_data_ptr);
//   // m_index.resize(n);
//   // m_index_bak.resize(n);
//   copy_from(other, m_size);
//   Logger::print_info("Finished copy constructor");
// }

template <typename ParticleClass>
particle_base<ParticleClass>::particle_base(
    particle_base<ParticleClass>&& other) {
  m_size = other.m_size;
  m_number = other.m_number;
  m_managed = other.m_managed;

  // m_data_ptr = other.m_data_ptr;
  m_data = other.m_data;
  m_tracked = other.m_tracked;
  m_tracked_ptc_map = other.m_tracked_ptc_map;
  // auto alloc = alloc_cuda_managed(m_size);
  // auto alloc = alloc_cuda_device(m_size);
  // alloc(m_index);
  cudaMalloc(&m_tmp_data_ptr, m_size * sizeof(double));
  cudaMalloc(&m_index, m_size * sizeof(Index_t));
  // alloc((double*)m_tmp_data_ptr);
  // m_index.resize(other.m_size);
  // m_index_bak.resize(other.m_size);

  // boost::fusion::for_each(other.m_data, set_nullptr());
  visit_struct::for_each(
      other.m_data, [](const char* name, auto& x) { x = nullptr; });
  visit_struct::for_each(
      other.m_tracked, [](const char* name, auto& x) { x = nullptr; });
  other.m_tracked_ptc_map = nullptr;
  // other.m_data_ptr = nullptr;
  // Logger::print_info("number is {}", m_number);
  // Logger::print_info("Finished move constructor");
}

template <typename ParticleClass>
particle_base<ParticleClass>::~particle_base() {
  free_mem();
  free_cuda()(m_index);
  free_cuda()(m_tmp_data_ptr);
  // CudaSafeCall(cudaFree(m_index));
  // CudaSafeCall(cudaFree(m_tmp_data_ptr));
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::alloc_mem(std::size_t max_num,
                                        bool managed,
                                        std::size_t alignment) {
  if (managed)
    alloc_struct_of_arrays_managed(m_data, max_num);
  else
    alloc_struct_of_arrays(m_data, max_num);
  // Tracked particles are always managed to make transferring between
  // device and host much easier
  m_max_tracked = std::min(max_num, size_t(MAX_TRACKED)); // No point in allocating more than max_num
  alloc_struct_of_arrays_managed(m_tracked, m_max_tracked);
  CudaSafeCall(
      cudaMalloc(&m_tracked_ptc_map, m_max_tracked * sizeof(uint32_t)));
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::free_mem() {
  free_struct_of_arrays(m_data);
  free_struct_of_arrays(m_tracked);
  CudaSafeCall(cudaFree(m_tracked_ptc_map));
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::initialize() {
  erase(0, m_size);
  m_number = 0;
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::resize(std::size_t max_num) {
  m_size = max_num;
  if (m_number > max_num) m_number = max_num;
  free_mem();
  alloc_mem(max_num);

  free_cuda()(m_index);
  free_cuda()(m_tmp_data_ptr);
  // auto alloc = alloc_cuda_managed(max_num);
  // auto alloc = alloc_cuda_device(max_num);
  // alloc(m_index);
  cudaMalloc(&m_index, max_num * sizeof(Index_t));
  cudaMalloc(&m_tmp_data_ptr, max_num * sizeof(double));
  // alloc((double*)m_tmp_data_ptr);

  // m_index.resize(max_num, 0);
  // m_index_bak.resize(max_num, 0);
  initialize();
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::erase(std::size_t pos,
                                    std::size_t amount) {
  if (pos + amount > m_size) amount = m_size - pos;
  // std::cout << "Erasing from index " << pos << " for " << amount
  //           << " number of particles" << std::endl;

  auto ptc = ParticleClass{};
  for_each_arg(m_data, ptc, fill_pos_amount{pos, amount});
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::put(Index_t pos,
                                  const ParticleClass& part) {
  if (pos >= m_size)
    throw std::runtime_error(
        "Trying to insert particle beyond the end of the array."
        "Resize it first!");

  for_each_arg(m_data, part, assign_at_idx(pos));
  if (pos >= m_number) m_number = pos + 1;
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::append(const ParticleClass& part) {
  // put(m_number, x, p, cell, flag);
  put(m_number, part);
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::copy_from(
    const particle_base<ParticleClass>& other, std::size_t num,
    std::size_t src_pos, std::size_t dest_pos) {
  // Adjust the number so that we don't over fill
  if (dest_pos + num > m_size) num = m_size - dest_pos;
  for_each_arg(m_data, other.m_data,
               copy_to_dest(num, src_pos, dest_pos));
  // Adjust the new number of particles in the array
  if (dest_pos + num > m_number) m_number = dest_pos + num;
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::copy_to_comm_buffers(
    std::vector<self_type>& buffers, const Quadmesh& mesh) {
  int num_buffers = buffers.size();
  // auto& mesh = grid.mesh();
  std::vector<int> num_ptc(num_buffers, 0);

  for (int i = 0; i < num_buffers; i++) {
    int zone = i;
    // if (zone >= num_buffers / 2) zone += 1;
    // Skip central buffer
    if (zone == (num_buffers - 1) / 2) continue;
    // Logger::print_info("copying to buffer {}", zone);
    if (mesh.dim() <= 2) zone += 9;
    if (mesh.dim() <= 1) zone += 3;

    auto compare_zone = [zone] __device__(uint32_t c) {
      return dev_mesh.find_zone(c) == zone;
    };

    visit_struct::for_each(
        m_data, buffers[i].m_data,
        [this, zone, &compare_zone, &num_ptc, i](const char* name,
                                                 auto& x, auto& y) {
          auto p_cell = thrust::device_pointer_cast(m_data.cell);
          auto p = thrust::device_pointer_cast(x);
          auto p_buf = thrust::device_pointer_cast(y);
          if (strcmp(name, "cell") == 0) {
            auto buf_end =
                thrust::copy_if(p, p + m_number, p_buf, compare_zone);
            num_ptc[i] = buf_end - p_buf;

            // TODO: adjust cell value after particles are copied to
            // communication buffer!!
            int dz = (zone / 9) - 1;
            int dy = (zone / 3) % 3 - 1;
            int dx = zone % 3 - 1;

            thrust::transform(p_buf, buf_end, p_buf, modify_cell{dx, dy, dz});
          } else {
            thrust::copy_if(p, p + m_number, p_cell, p_buf,
                            compare_zone);
          }
        });
    // Logger::print_info("setting buffer {} number to {}", i,
    // num_ptc[i]);
    buffers[i].set_num(num_ptc[i]);
  }
  // Now that all outgoing particles are in buffers, set them to empty
  // in the main array
  auto p_cell = thrust::device_pointer_cast(m_data.cell);
  thrust::replace_if(
      p_cell, p_cell + m_number,
      [] __device__(uint32_t c) { return dev_mesh.find_zone(c) != 13; },
      MAX_CELL);
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::sort_by_cell(const Grid& grid) {
  if (m_number > 0) {
    // Generate particle index array
    auto ptr_cell = thrust::device_pointer_cast(m_data.cell);
    auto ptr_idx = thrust::device_pointer_cast(m_index);
    thrust::counting_iterator<Index_t> iter(0);
    thrust::copy_n(iter, m_number, ptr_idx);

    // Sort the index array by key
    thrust::sort_by_key(ptr_cell, ptr_cell + m_number, ptr_idx);
    // cudaDeviceSynchronize();
    Logger::print_debug("Finished sorting");

    // Move the rest of particle array using the new index
    rearrange_arrays("cell");

    // Update the new number of particles
    const int padding = 100;
    m_number =
        thrust::upper_bound(ptr_cell, ptr_cell + m_number + padding,
                            MAX_CELL - 1) -
        ptr_cell;

    // Logger::print_info("Sorting complete, there are {} particles in
    // the pool", m_number); cudaDeviceSynchronize();
    CudaCheckError();
  }
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::rearrange_arrays(
    const std::string& skip) {
  const uint32_t padding = 100;
  auto ptc = ParticleClass();
  for_each_arg_with_name(
      m_data, ptc,
      [this, &padding, &skip](const char* name, auto& x, auto& u) {
        typedef
            typename std::remove_reference<decltype(x)>::type x_type;
        auto ptr_index = thrust::device_pointer_cast(m_index);
        if (std::strcmp(name, skip.c_str()) == 0) return;

        auto x_ptr = thrust::device_pointer_cast(x);
        auto tmp_ptr = thrust::device_pointer_cast(
            reinterpret_cast<x_type>(m_tmp_data_ptr));
        thrust::gather(ptr_index, ptr_index + m_number, x_ptr, tmp_ptr);
        thrust::copy_n(tmp_ptr, m_number, x_ptr);
        CudaCheckError();
      });
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::clear_guard_cells(const Grid& grid) {
  erase_ptc_in_guard_cells(m_data.cell, m_number);
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::get_tracked_ptc() {
  if (m_number > 0) {
    uint32_t* num_tracked;
    uint32_t init_num_tracked = 0;
    CudaSafeCall(cudaMalloc(&num_tracked, sizeof(uint32_t)));
    // CudaSafeCall(cudaMemset(num_tracked, 0, sizeof(uint32_t)));
    CudaSafeCall(cudaMemcpy(num_tracked, &init_num_tracked,
                            sizeof(uint32_t), cudaMemcpyHostToDevice));

    map_tracked_ptc(m_data.flag, m_data.cell, m_number,
                    m_tracked_ptc_map, num_tracked, m_max_tracked);
    CudaSafeCall(cudaMemcpy(&m_num_tracked, num_tracked,
                            sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaFree(num_tracked));

    if (m_num_tracked >= m_max_tracked) m_num_tracked = m_max_tracked - 1;
    visit_struct::for_each(
        m_data, m_tracked, [this](const char* name, auto& u, auto& v) {
          Kernels::get_tracked_ptc_attr<<<256, 512>>>(
              u, m_tracked_ptc_map, m_num_tracked, v);
          CudaCheckError();
        });
    CudaSafeCall(cudaDeviceSynchronize());
    Logger::print_info("Got {} tracked particles", m_num_tracked);
  }
}

template <typename ParticleClass>
void
particle_base<ParticleClass>::get_total_and_offset(uint64_t num) {
  // Carry out an MPI scan to get the total number and local offset,
  // used for particle output into a file
  uint64_t result = 0;
  auto status =
      MPI_Scan(&num, &result, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  uint64_t offset = result - num;
  uint64_t total = 0;
  status = MPI_Allreduce(&num, &total, 1, MPI_UINT64_T, MPI_SUM,
                         MPI_COMM_WORLD);
  m_offset = offset;
  m_total = total;
}

// template <typename ParticleClass>
// void
// particle_base<ParticleClass>::compute_spectrum(
//     int num_bins, std::vector<Scalar>& energies,
//     std::vector<uint32_t>& nums) {
//   // Assume the particle energies have been computed
//   energies.resize(num_bins, 0.0);
//   nums.resize(num_bins, 0);

//   // Find maximum energy in the array now
//   thrust::device_ptr<Scalar> E_ptr =
//       thrust::device_pointer_cast(m_data.E);
//   Scalar E_max = *thrust::max_element(E_ptr, E_ptr + m_number);
//   // Logger::print_info("Maximum energy is {}", E_max);

//   // Partition the energy bin up to max energy times a factor
//   Scalar dlogE = std::log(E_max) / (Scalar)num_bins;
//   for (int i = 0; i < num_bins; i++) {
//     energies[i] = std::exp((0.5f + (Scalar)i) * dlogE);
//     // Logger::print_info("{}", energies[i]);
//   }

//   // Do a histogram
//   uint32_t* d_energies;
//   cudaMalloc(&d_energies, num_bins * sizeof(uint32_t));
//   thrust::device_ptr<uint32_t> ptr_energies =
//       thrust::device_pointer_cast(d_energies);
//   thrust::fill_n(ptr_energies, num_bins, 0);
//   // cudaDeviceSynchronize();

//   compute_energy_histogram(d_energies, m_data.E, m_number, num_bins,
//                            E_max);

//   // Copy the resulting histogram to output
//   cudaMemcpy(nums.data(), d_energies, num_bins * sizeof(uint32_t),
//              cudaMemcpyDeviceToHost);

//   cudaFree(d_energies);
// }

}  // namespace Aperture

#endif  // _PARTICLE_BASE_IMPL_CUH_
