#include "core/detail/multi_array_utils.hpp"
#include "core/interpolation.h"
#include "cuda/constant_mem.h"
#include "cuda/core/field_solver_helper.cuh"
#include "cuda/cudaUtility.h"
#include "cuda/data/fields_dev.h"

namespace Aperture {

namespace Kernels {

template <int Order, int DIM1, int DIM2, int DIM3>
__global__ void
interp_to_center(cudaPitchedPtr v1, cudaPitchedPtr v2,
                 cudaPitchedPtr v3, cudaPitchedPtr u1,
                 cudaPitchedPtr u2, cudaPitchedPtr u3, FieldType type,
                 Scalar q = 1.0) {
  // Declare cache array in shared memory
  __shared__ Scalar
      s_u1[DIM3 + 2 * Pad<Order>::val][DIM2 + 2 * Pad<Order>::val]
          [DIM1 + 2 * Pad<Order>::val];
  __shared__ Scalar
      s_u2[DIM3 + 2 * Pad<Order>::val][DIM2 + 2 * Pad<Order>::val]
          [DIM1 + 2 * Pad<Order>::val];
  __shared__ Scalar
      s_u3[DIM3 + 2 * Pad<Order>::val][DIM2 + 2 * Pad<Order>::val]
          [DIM1 + 2 * Pad<Order>::val];

  // Load indices
  int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x + Pad<Order>::val,
      c2 = threadIdx.y + Pad<Order>::val,
      c3 = threadIdx.z + Pad<Order>::val;
  size_t globalOffset =
      +(dev_mesh.guard[2] + t3 * DIM3 + c3 - Pad<Order>::val) *
          u1.pitch * u1.ysize +
      (dev_mesh.guard[1] + t2 * DIM2 + c2 - Pad<Order>::val) *
          u1.pitch +
      (dev_mesh.guard[0] + t1 * DIM1 + c1 - Pad<Order>::val) *
          sizeof(Scalar);

  // Load shared memory
  init_shared_memory<Order, DIM1, DIM2, DIM3>(
      s_u1, s_u2, s_u3, u1, u2, u3, globalOffset, c1, c2, c3);
  __syncthreads();

  if (type == FieldType::E) {
    (*(Scalar*)((char*)v1.ptr + globalOffset)) +=
        q * 0.5f * (s_u1[c3][c2][c1] + s_u1[c3][c2][c1 - 1]);
    (*(Scalar*)((char*)v2.ptr + globalOffset)) +=
        q * 0.5f * (s_u2[c3][c2][c1] + s_u2[c3][c2 - 1][c1]);
    (*(Scalar*)((char*)v3.ptr + globalOffset)) +=
        q * 0.5f * (s_u3[c3][c2][c1] + s_u3[c3 - 1][c2][c1]);
  } else {
    (*(Scalar*)((char*)v1.ptr + globalOffset)) +=
        q * 0.25f *
        (s_u1[c3][c2][c1] + s_u1[c3 - 1][c2][c1] +
         s_u1[c3][c2 - 1][c1] + s_u1[c3 - 1][c2 - 1][c1]);
    (*(Scalar*)((char*)v2.ptr + globalOffset)) +=
        q * 0.25f *
        (s_u2[c3][c2][c1] + s_u2[c3 - 1][c2][c1] +
         s_u2[c3][c2][c1 - 1] + s_u2[c3 - 1][c2][c1 - 1]);
    (*(Scalar*)((char*)v3.ptr + globalOffset)) +=
        q * 0.25f *
        (s_u3[c3][c2][c1] + s_u3[c3][c2][c1 - 1] +
         s_u3[c3][c2 - 1][c1] + s_u3[c3][c2 - 1][c1 - 1]);
  }
}

template <int Order, int DIM1, int DIM2, int DIM3>
__global__ void
interp_from_center(cudaPitchedPtr v1, cudaPitchedPtr v2,
                   cudaPitchedPtr v3, cudaPitchedPtr u1,
                   cudaPitchedPtr u2, cudaPitchedPtr u3, FieldType type,
                   Scalar q = 1.0) {
  // Declare cache array in shared memory
  __shared__ Scalar
      s_u1[DIM3 + 2 * Pad<Order>::val][DIM2 + 2 * Pad<Order>::val]
          [DIM1 + 2 * Pad<Order>::val];
  __shared__ Scalar
      s_u2[DIM3 + 2 * Pad<Order>::val][DIM2 + 2 * Pad<Order>::val]
          [DIM1 + 2 * Pad<Order>::val];
  __shared__ Scalar
      s_u3[DIM3 + 2 * Pad<Order>::val][DIM2 + 2 * Pad<Order>::val]
          [DIM1 + 2 * Pad<Order>::val];

  // Load indices
  // int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x + Pad<Order>::val,
      c2 = threadIdx.y + Pad<Order>::val,
      c3 = threadIdx.z + Pad<Order>::val;
  size_t globalOffset =
      +(dev_mesh.guard[2] + blockIdx.z * DIM3 + c3 - Pad<Order>::val) *
          u1.pitch * u1.ysize +
      (dev_mesh.guard[1] + blockIdx.y * DIM2 + c2 - Pad<Order>::val) *
          u1.pitch +
      (dev_mesh.guard[0] + blockIdx.x * DIM1 + c1 - Pad<Order>::val) *
          sizeof(Scalar);

  // Load shared memory
  init_shared_memory<Order, DIM1, DIM2, DIM3>(
      s_u1, s_u2, s_u3, u1, u2, u3, globalOffset, c1, c2, c3);
  __syncthreads();

  // if (type == FieldType::E) {
  (*(Scalar*)((char*)v1.ptr + globalOffset)) +=
      q * 0.5f * (s_u1[c3][c2][c1 + 1] + s_u1[c3][c2][c1]);
  (*(Scalar*)((char*)v2.ptr + globalOffset)) +=
      q * 0.5f * (s_u2[c3][c2 + 1][c1] + s_u2[c3][c2][c1]);
  (*(Scalar*)((char*)v3.ptr + globalOffset)) +=
      q * 0.5f * (s_u3[c3 + 1][c2][c1] + s_u3[c3][c2][c1]);
  // } else {
  //   (*(Scalar*)((char*)v1.ptr + globalOffset)) +=
  //       q * 0.25f * (s_u1[c3][c2][c1] + s_u1[c3 + 1][c2][c1] +
  //                s_u1[c3][c2 + 1][c1] + s_u1[c3 + 1][c2 + 1][c1]);
  //   (*(Scalar*)((char*)v2.ptr + globalOffset)) +=
  //       q * 0.25f * (s_u2[c3][c2][c1] + s_u2[c3 + 1][c2][c1] +
  //                s_u2[c3][c2][c1 + 1] + s_u2[c3 + 1][c2][c1 + 1]);
  //   (*(Scalar*)((char*)v3.ptr + globalOffset)) +=
  //       q * 0.25f * (s_u3[c3][c2][c1] + s_u3[c3][c2][c1 + 1] +
  //                s_u3[c3][c2 + 1][c1] + s_u3[c3][c2 + 1][c1 + 1]);
  // }
}

template <int Order, int DIM1, int DIM2>
__global__ void
interp_from_center_add_2d(cudaPitchedPtr v1, cudaPitchedPtr v2,
                          cudaPitchedPtr v3, cudaPitchedPtr u1,
                          cudaPitchedPtr u2, cudaPitchedPtr u3,
                          FieldType type, Scalar q = 1.0) {
  // Declare cache array in shared memory
  __shared__ Scalar
      s_u1[DIM2 + 2 * Pad<Order>::val][DIM1 + 2 * Pad<Order>::val];
  __shared__ Scalar
      s_u2[DIM2 + 2 * Pad<Order>::val][DIM1 + 2 * Pad<Order>::val];
  __shared__ Scalar
      s_u3[DIM2 + 2 * Pad<Order>::val][DIM1 + 2 * Pad<Order>::val];

  // Load indices
  // int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x + Pad<Order>::val,
      c2 = threadIdx.y + Pad<Order>::val;
  size_t globalOffset =
      (dev_mesh.guard[1] + blockIdx.y * DIM2 + c2 - Pad<Order>::val) *
          u1.pitch +
      (dev_mesh.guard[0] + blockIdx.x * DIM1 + c1 - Pad<Order>::val) *
          sizeof(Scalar);

  // Load shared memory
  init_shared_memory_2d<Order, DIM1, DIM2>(s_u1, s_u2, s_u3, u1, u2, u3,
                                           globalOffset, c1, c2);
  __syncthreads();

  // if (type == FieldType::E) {
  (*(Scalar*)((char*)v1.ptr + globalOffset)) +=
      q * 0.5f * (s_u1[c2][c1 + 1] + s_u1[c2][c1]);
  (*(Scalar*)((char*)v2.ptr + globalOffset)) +=
      q * 0.5f * (s_u2[c2 + 1][c1] + s_u2[c2][c1]);
  (*(Scalar*)((char*)v3.ptr + globalOffset)) += q * s_u3[c2][c1];
  // } else {
  //   (*(Scalar*)((char*)v1.ptr + globalOffset)) +=
  //       q * 0.25f * (s_u1[c3][c2][c1] + s_u1[c3 + 1][c2][c1] +
  //                s_u1[c3][c2 + 1][c1] + s_u1[c3 + 1][c2 + 1][c1]);
  //   (*(Scalar*)((char*)v2.ptr + globalOffset)) +=
  //       q * 0.25f * (s_u2[c3][c2][c1] + s_u2[c3 + 1][c2][c1] +
  //                s_u2[c3][c2][c1 + 1] + s_u2[c3 + 1][c2][c1 + 1]);
  //   (*(Scalar*)((char*)v3.ptr + globalOffset)) +=
  //       q * 0.25f * (s_u3[c3][c2][c1] + s_u3[c3][c2][c1 + 1] +
  //                s_u3[c3][c2 + 1][c1] + s_u3[c3][c2 + 1][c1 + 1]);
  // }
}

template <int Order, int DIM1, int DIM2>
__global__ void
interp_from_center_2d(cudaPitchedPtr v1, cudaPitchedPtr v2,
                      cudaPitchedPtr v3, cudaPitchedPtr u1,
                      cudaPitchedPtr u2, cudaPitchedPtr u3,
                      FieldType type, Scalar q = 1.0) {
  // Declare cache array in shared memory
  __shared__ Scalar
      s_u1[DIM2 + 2 * Pad<Order>::val][DIM1 + 2 * Pad<Order>::val];
  __shared__ Scalar
      s_u2[DIM2 + 2 * Pad<Order>::val][DIM1 + 2 * Pad<Order>::val];
  __shared__ Scalar
      s_u3[DIM2 + 2 * Pad<Order>::val][DIM1 + 2 * Pad<Order>::val];

  // Load indices
  // int t1 = blockIdx.x, t2 = blockIdx.y, t3 = blockIdx.z;
  int c1 = threadIdx.x + Pad<Order>::val,
      c2 = threadIdx.y + Pad<Order>::val;
  size_t globalOffset =
      (dev_mesh.guard[1] + blockIdx.y * DIM2 + c2 - Pad<Order>::val) *
          u1.pitch +
      (dev_mesh.guard[0] + blockIdx.x * DIM1 + c1 - Pad<Order>::val) *
          sizeof(Scalar);

  // Load shared memory
  init_shared_memory_2d<Order, DIM1, DIM2>(s_u1, s_u2, s_u3, u1, u2, u3,
                                           globalOffset, c1, c2);
  __syncthreads();

  // if (type == FieldType::E) {
  (*(Scalar*)((char*)v1.ptr + globalOffset)) =
      q * 0.5f * (s_u1[c2][c1 + 1] + s_u1[c2][c1]);
  (*(Scalar*)((char*)v2.ptr + globalOffset)) =
      q * 0.5f * (s_u2[c2 + 1][c1] + s_u2[c2][c1]);
  (*(Scalar*)((char*)v3.ptr + globalOffset)) = q * s_u3[c2][c1];
  // } else {
  //   (*(Scalar*)((char*)v1.ptr + globalOffset)) +=
  //       q * 0.25f * (s_u1[c3][c2][c1] + s_u1[c3 + 1][c2][c1] +
  //                s_u1[c3][c2 + 1][c1] + s_u1[c3 + 1][c2 + 1][c1]);
  //   (*(Scalar*)((char*)v2.ptr + globalOffset)) +=
  //       q * 0.25f * (s_u2[c3][c2][c1] + s_u2[c3 + 1][c2][c1] +
  //                s_u2[c3][c2][c1 + 1] + s_u2[c3 + 1][c2][c1 + 1]);
  //   (*(Scalar*)((char*)v3.ptr + globalOffset)) +=
  //       q * 0.25f * (s_u3[c3][c2][c1] + s_u3[c3][c2][c1 + 1] +
  //                s_u3[c3][c2 + 1][c1] + s_u3[c3][c2 + 1][c1 + 1]);
  // }
}

}  // namespace Kernels

////////////////////////////////////////////////////////////////////////////////
//  Scalar Field Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
cu_scalar_field<T>::cu_scalar_field() : field_base(), m_array() {}

template <typename T>
cu_scalar_field<T>::cu_scalar_field(const grid_type& grid,
                                    Stagger stagger)
    : field_base(grid), m_array(grid.extent()), m_stagger(stagger) {
  // std::cout << grid.extent() << std::endl;
  m_stagger = Stagger(0b111);
}

template <typename T>
cu_scalar_field<T>::cu_scalar_field(const self_type& field)
    : field_base(*field.m_grid),
      m_array(field.m_array),
      m_stagger(field.m_stagger) {}

template <typename T>
cu_scalar_field<T>::cu_scalar_field(self_type&& field)
    : field_base(*field.m_grid),
      m_array(std::move(field.m_array)),
      m_stagger(field.m_stagger) {}

template <typename T>
cu_scalar_field<T>::~cu_scalar_field() {}

template <typename T>
void
cu_scalar_field<T>::initialize() {
  // Assign the field to zero, whatever 0 corresponds to for type #T
  m_array.assign_dev(static_cast<T>(0));
}

template <typename T>
void
cu_scalar_field<T>::assign(data_type value) {
  // Assign a uniform value to the array
  m_array.assign_dev(value);
}

template <typename T>
void
cu_scalar_field<T>::copy_from(const self_type& field) {
  /// We can copy as long as the extents are the same
  this->check_grid_extent(this->m_grid->extent(),
                          field.grid().extent());

  m_array.copy_from(field.m_array);
}

template <typename T>
cu_scalar_field<T>&
cu_scalar_field<T>::operator=(const self_type& field) {
  this->m_grid = field.m_grid;
  this->m_grid_size = field.m_grid_size;
  m_array = field.m_array;
  return (*this);
}

template <typename T>
cu_scalar_field<T>&
cu_scalar_field<T>::operator=(self_type&& field) {
  this->m_grid = field.m_grid;
  this->m_grid_size = field.m_grid_size;
  m_array = std::move(field.m_array);
  return (*this);
}

template <typename T>
void
cu_scalar_field<T>::resize(const Grid& grid) {
  this->m_grid = &grid;
  this->m_grid_size = grid.size();
  m_array.resize(grid.extent(), m_array.devId());
}

template <typename T>
cu_scalar_field<T>&
cu_scalar_field<T>::multiplyBy(data_type value) {
  // detail::map_multi_array(m_array.begin(), this -> m_grid ->
  // extent(), detail::Op_MultConst<T>(value));
  dim3 blockSize(8, 8, 8);
  dim3 gridSize(8, 8, 8);
  Kernels::map_array_unary_op<T>
      <<<gridSize, blockSize>>>(m_array.data_d(), m_grid->extent(),
                                detail::Op_MultConst<T>(value));
  return (*this);
}

template <typename T>
cu_scalar_field<T>&
cu_scalar_field<T>::multiplyBy(const cu_scalar_field<T>& field) {
  this->check_grid_extent(this->m_grid->extent(),
                          field.grid().extent());

  // detail::map_multi_array(m_array.begin(), field.data().begin(), this
  // -> m_grid -> extent(),
  //                         detail::Op_MultAssign<T>());
  dim3 blockSize(8, 8, 8);
  dim3 gridSize(8, 8, 8);
  Kernels::map_array_binary_op<T><<<gridSize, blockSize>>>(
      field.data().data_d(), m_array.data_d(), m_grid->extent(),
      detail::Op_MultAssign<T>());
  return (*this);
}

template <typename T>
cu_scalar_field<T>&
cu_scalar_field<T>::addBy(data_type value) {
  // detail::map_multi_array(m_array.begin(), this -> m_grid ->
  // extent(), detail::Op_PlusConst<T>(value));
  dim3 blockSize(8, 8, 8);
  dim3 gridSize(8, 8, 8);
  Kernels::map_array_unary_op<T>
      <<<gridSize, blockSize>>>(m_array.data_d(), m_grid->extent(),
                                detail::Op_PlusConst<T>(value));
  return (*this);
}

template <typename T>
cu_scalar_field<T>&
cu_scalar_field<T>::addBy(const cu_scalar_field<T>& field) {
  this->check_grid_extent(this->m_grid->extent(),
                          field.grid().extent());

  dim3 blockSize(8, 8, 8);
  dim3 gridSize(8, 8, 8);
  // detail::map_multi_array(m_array.begin(), field.data().begin(), this
  // -> m_grid -> extent(),
  //                         detail::Op_PlusAssign<T>());
  Kernels::map_array_binary_op<T><<<gridSize, blockSize>>>(
      field.data().data_d(), m_array.data_d(), m_grid->extent(),
      detail::Op_PlusAssign<T>());
  return (*this);
}

template <typename T>
cu_scalar_field<T>&
cu_scalar_field<T>::subtractBy(data_type value) {
  dim3 blockSize(8, 8, 8);
  dim3 gridSize(8, 8, 8);
  // detail::map_multi_array(m_array.begin(), this -> m_grid ->
  // extent(), detail::Op_MinusConst<T>(value));
  Kernels::map_array_unary_op<T>
      <<<gridSize, blockSize>>>(m_array.data_d(), m_grid->extent(),
                                detail::Op_MinusConst<T>(value));
  return (*this);
}

template <typename T>
cu_scalar_field<T>&
cu_scalar_field<T>::subtractBy(const cu_scalar_field<T>& field) {
  this->check_grid_extent(this->m_grid->extent(),
                          field.grid().extent());

  // detail::map_multi_array(m_array.begin(), field.data().begin(), this
  // -> m_grid -> extent(),
  //                         detail::Op_MinusAssign<T>());
  dim3 blockSize(8, 8, 8);
  dim3 gridSize(8, 8, 8);
  Kernels::map_array_binary_op<T><<<gridSize, blockSize>>>(
      field.data().data_d(), m_array.data_d(), m_grid->extent(),
      detail::Op_MinusAssign<T>());
  return (*this);
}

template <typename T>
cu_scalar_field<T>&
cu_scalar_field<T>::divideBy(const cu_scalar_field<T>& field) {
  this->check_grid_extent(this->m_grid->extent(),
                          field.grid().extent());

  if (field.grid().dim() == 3) {
    dim3 gridSize(8, 8, 8);
    dim3 blockSize(8, 8, 8);
    Kernels::map_array_binary_op<T><<<gridSize, blockSize>>>(
        field.data().data_d(), m_array.data_d(), m_grid->extent(),
        detail::Op_DivAssign<T>());
  } else if (field.grid().dim() == 2) {
    dim3 gridSize(32, 32);
    dim3 blockSize(32, 32);
    Kernels::map_array_binary_op_2d<T><<<gridSize, blockSize>>>(
        field.data().data_d(), m_array.data_d(), m_grid->extent(),
        detail::Op_DivAssign<T>());
  }
  CudaCheckError();
  return (*this);
}

template <typename T>
// template <int Order>
T
cu_scalar_field<T>::interpolate(const Vec3<int>& c,
                                const Vec3<Pos_t>& rel_pos,
                                int order) const {
  Interpolator interp(order);
  // Vec3<int> c = m_grid -> mesh().getCell(cell);
  Vec3<int> lower = c - interp.radius();
  Vec3<int> upper = c + interp.support() - interp.radius();
  if (m_grid->dim() < 3) {
    lower[2] = upper[2] = c[2];
  }
  T result{};
  for (int k = lower[2]; k <= upper[2]; k++) {
    for (int j = lower[1]; j <= upper[1]; j++) {
      for (int i = lower[0]; i <= upper[0]; i++) {
        if (m_grid->dim() < 3) {
          result +=
              m_array(i, j, k) *
              interp.interp_cell(rel_pos[0], c[0], i, m_stagger[0]) *
              interp.interp_cell(rel_pos[1], c[1], j, m_stagger[1]);
        } else {
          result += m_array(i, j, k) *
                    interp.interp_weight(rel_pos, c, Vec3<int>(i, j, k),
                                         m_stagger);
        }
      }
    }
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////
//  Vector Field Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename T>
cu_vector_field<T>::cu_vector_field() : field_base(), m_array() {
  for (int i = 0; i < VECTOR_DIM; ++i) {
    // Default initialize to face-centered
    m_stagger[i] = Stagger(0b000);
    // m_stagger[i][i] = true;
    m_stagger[i].set_bit(i, true);
  }
}

template <typename T>
cu_vector_field<T>::cu_vector_field(const grid_type& grid)
    : field_base(grid) {
  m_type = FieldType::E;
  for (int i = 0; i < VECTOR_DIM; ++i) {
    m_array[i] = array_type(grid.extent());
    // Default initialize to face-centered
    m_stagger[i] = Stagger();
    // m_stagger[i].set_bit(i, true);
    m_stagger[i].set_bit((i + 1) % 3, true);
    m_stagger[i].set_bit((i + 2) % 3, true);
  }
  // init_array_ptrs();
}

template <typename T>
cu_vector_field<T>::cu_vector_field(const self_type& field)
    : field_base(*field.m_grid),
      m_array(field.m_array),
      m_stagger(field.m_stagger),
      m_type(field.m_type) {
  // init_array_ptrs();
}

template <typename T>
cu_vector_field<T>::cu_vector_field(self_type&& field)
    : field_base(*field.m_grid),
      m_array(std::move(field.m_array)),
      m_stagger(field.m_stagger),
      m_type(field.m_type) {
  // init_array_ptrs();
}

template <typename T>
cu_vector_field<T>::~cu_vector_field() {
  // cudaFree(m_ptrs);
}

template <typename T>
cu_vector_field<T>&
cu_vector_field<T>::operator=(const self_type& other) {
  this->m_grid = other.m_grid;
  this->m_grid_size = other.m_grid_size;
  m_array = other.m_array;
  m_type = other.m_type;
  // for (int i = 0; i < VECTOR_DIM; ++i) {
  //   m_ptrs[i] = m_array[i].data();
  // }

  return (*this);
}

template <typename T>
cu_vector_field<T>&
cu_vector_field<T>::operator=(self_type&& other) {
  this->m_grid = other.m_grid;
  this->m_grid_size = other.m_grid_size;
  m_array = std::move(other.m_array);
  m_type = other.m_type;
  // for (int i = 0; i < VECTOR_DIM; ++i) {
  //   m_ptrs[i] = m_array[i].data();
  // }

  return (*this);
}

template <typename T>
void
cu_vector_field<T>::initialize() {
  for (int i = 0; i < VECTOR_DIM; ++i) {
    m_array[i].assign_dev(static_cast<T>(0));
  }
}

// template <typename T>
// void
// cu_vector_field<T>::init_array_ptrs() {
//   // CudaSafeCall(cudaMallocManaged(&m_ptrs, VECTOR_DIM *
//   sizeof(T*)));

//   // for (int i = 0; i < VECTOR_DIM; ++i) {
//   //   m_ptrs[i] = m_array[i].data();
//   // }
// }

// template <typename T>
// void vector_field<T>::initialize(const initial_condition& ic,
// FieldType type) {
//   for (int component = 0; component < 3; component ++) {
//     // This way vector field is always defined in the center of the
//     cell
//     // face, staggered in the direction of the component
//     for (int k = 0; k < m_grid -> extent().depth(); ++k) {
//       double x3 = m_grid -> mesh().pos(2, k, component == 2 ? 1 : 0);
//       for (int j = 0; j < m_grid -> extent().height(); ++j) {
//         double x2 = m_grid -> mesh().pos(1, j, component == 1 ? 1 :
//         0); for (int i = 0; i < m_grid -> extent().width(); ++i) {
//           double x1 = m_grid -> mesh().pos(0, i, component == 0 ? 1 :
//           0);
//           // Stagger is automatically taken care of by the grid
//           if (type == FieldType::E)
//             m_array[component](i, j, k) = ic.E(component, x1, x2,
//             x3);
//           else if (type == FieldType::B)
//             m_array[component](i, j, k) = ic.B(component, x1, x2,
//             x3);
//           if (type == FieldType::B && i == 20 && j == m_grid ->
//           mesh().dims[1] - m_grid -> mesh().guard[1] - 1)
//             std::cout << Vec3<double>(x1, x2, x3) << std::endl;
//         }
//       }
//     }
//   }
// }

template <typename T>
void
cu_vector_field<T>::assign(data_type value, int n) {
  m_array[n].assign_dev(value);
}

template <typename T>
void
cu_vector_field<T>::assign(data_type value) {
  for (int i = 0; i < VECTOR_DIM; i++) {
    m_array[i].assign_dev(value);
  }
}

template <typename T>
void
cu_vector_field<T>::assign(const cu_vector_field<T>& field, T q) {
  this->check_grid_extent(this->m_grid->extent(),
                          field.grid().extent());

  if (field.grid().dim() == 3) {
    dim3 gridSize(8, 8, 8);
    dim3 blockSize(8, 8, 8);
    for (int i = 0; i < VECTOR_DIM; ++i) {
      // detail::map_multi_array(m_array[i].begin(),
      // field.data(i).begin(),
      //                         this -> m_grid -> extent(),
      //                         detail::Op_PlusAssign<T>());
      Kernels::map_array_binary_op<T><<<gridSize, blockSize>>>(
          field.data(i).data_d(), m_array[i].data_d(), m_grid->extent(),
          detail::Op_AddMultConst<T>(q));
    }
  } else if (field.grid().dim() == 2) {
    dim3 gridSize(32, 32);
    dim3 blockSize(32, 32);
    for (int i = 0; i < VECTOR_DIM; ++i) {
      Kernels::map_array_binary_op_2d<T><<<gridSize, blockSize>>>(
          field.data(i).data_d(), m_array[i].data_d(), m_grid->extent(),
          detail::Op_AddMultConst<T>(q));
    }
  }
  // return (*this);
}

template <typename T>
void
cu_vector_field<T>::copy_from(const self_type& field) {
  /// We can copy as long as the extents are the same
  this->check_grid_extent(this->m_grid->extent(),
                          field.grid().extent());

  for (int i = 0; i < VECTOR_DIM; ++i) {
    m_array[i].copy_from(field.m_array[i]);
  }
}

template <typename T>
void
cu_vector_field<T>::resize(const Grid& grid) {
  this->m_grid = &grid;
  this->m_grid_size = grid.size();
  for (int i = 0; i < VECTOR_DIM; i++) {
    m_array[i].resize(grid.extent());
  }
}

template <typename T>
cu_vector_field<T>&
cu_vector_field<T>::multiplyBy(data_type value) {
  dim3 gridSize(8, 8, 8);
  dim3 blockSize(8, 8, 8);
  for (int i = 0; i < VECTOR_DIM; ++i) {
    // detail::map_multi_array(m_array[i].begin(), this -> m_grid ->
    // extent(),
    //                         detail::Op_MultConst<T>(value));
    Kernels::map_array_unary_op<T>
        <<<gridSize, blockSize>>>(m_array[i].data_d(), m_grid->extent(),
                                  detail::Op_MultConst<T>(value));
  }
  return (*this);
}

template <typename T>
cu_vector_field<T>&
cu_vector_field<T>::multiplyBy(const cu_scalar_field<T>& field) {
  this->check_grid_extent(this->m_grid->extent(),
                          field.grid().extent());

  dim3 gridSize(8, 8, 8);
  dim3 blockSize(8, 8, 8);
  for (int i = 0; i < VECTOR_DIM; ++i) {
    Kernels::map_array_binary_op<T><<<gridSize, blockSize>>>(
        field.data().data_d(), m_array[i].data_d(), m_grid->extent(),
        detail::Op_MultAssign<T>());
    // detail::map_multi_array(m_array[i].begin(), field.data().begin(),
    //                         this -> m_grid -> extent(),
    //                         detail::Op_MultAssign<T>());
  }
  return (*this);
}

template <typename T>
cu_vector_field<T>&
cu_vector_field<T>::addBy(data_type value, int n) {
  // detail::map_multi_array(m_array[n].begin(), this -> m_grid ->
  // extent(),
  //                         detail::Op_PlusConst<T>(value));
  if (this->m_grid->dim() == 3) {
    dim3 gridSize(8, 8, 8);
    dim3 blockSize(8, 8, 8);
    Kernels::map_array_unary_op<T>
        <<<gridSize, blockSize>>>(m_array[n].data_d(), m_grid->extent(),
                                  detail::Op_PlusConst<T>(value));
  } else if (this->m_grid->dim() == 2) {
    dim3 gridSize(32, 32);
    dim3 blockSize(32, 32);
    Kernels::map_array_unary_op_2d<T>
        <<<gridSize, blockSize>>>(m_array[n].data_d(), m_grid->extent(),
                                  detail::Op_PlusConst<T>(value));
  }
  return (*this);
}

template <typename T>
cu_vector_field<T>&
cu_vector_field<T>::addBy(const cu_vector_field<T>& field) {
  this->check_grid_extent(this->m_grid->extent(),
                          field.grid().extent());

  if (field.grid().dim() == 3) {
    dim3 gridSize(8, 8, 8);
    dim3 blockSize(8, 8, 8);
    for (int i = 0; i < VECTOR_DIM; ++i) {
      // detail::map_multi_array(m_array[i].begin(),
      // field.data(i).begin(),
      //                         this -> m_grid -> extent(),
      //                         detail::Op_PlusAssign<T>());
      Kernels::map_array_binary_op<T><<<gridSize, blockSize>>>(
          field.data(i).data_d(), m_array[i].data_d(), m_grid->extent(),
          detail::Op_PlusAssign<T>());
    }
  } else if (field.grid().dim() == 2) {
    dim3 gridSize(32, 32);
    dim3 blockSize(32, 32);
    for (int i = 0; i < VECTOR_DIM; ++i) {
      Kernels::map_array_binary_op_2d<T><<<gridSize, blockSize>>>(
          field.data(i).data_d(), m_array[i].data_d(), m_grid->extent(),
          detail::Op_PlusAssign<T>());
    }
  }
  return (*this);
}

template <typename T>
cu_vector_field<T>&
cu_vector_field<T>::addBy(const cu_vector_field<T>& field, T q) {
  this->check_grid_extent(this->m_grid->extent(),
                          field.grid().extent());

  if (field.grid().dim() == 3) {
    dim3 gridSize(8, 8, 8);
    dim3 blockSize(8, 8, 8);
    for (int i = 0; i < VECTOR_DIM; ++i) {
      // detail::map_multi_array(m_array[i].begin(),
      // field.data(i).begin(),
      //                         this -> m_grid -> extent(),
      //                         detail::Op_PlusAssign<T>());
      Kernels::map_array_binary_op<T><<<gridSize, blockSize>>>(
          field.data(i).data_d(), m_array[i].data_d(), m_grid->extent(),
          detail::Op_MultConstAdd<T>(q));
    }
  } else if (field.grid().dim() == 2) {
    dim3 gridSize(32, 32);
    dim3 blockSize(32, 32);
    for (int i = 0; i < VECTOR_DIM; ++i) {
      Kernels::map_array_binary_op_2d<T><<<gridSize, blockSize>>>(
          field.data(i).data_d(), m_array[i].data_d(), m_grid->extent(),
          detail::Op_MultConstAdd<T>(q));
    }
  }
  return (*this);
}

template <typename T>
cu_vector_field<T>&
cu_vector_field<T>::subtractBy(data_type value, int n) {
  // detail::map_multi_array(m_array[n].begin(), this -> m_grid ->
  // extent(),
  //                         detail::Op_MinusConst<T>(value));
  dim3 gridSize(8, 8, 8);
  dim3 blockSize(8, 8, 8);
  Kernels::map_array_unary_op<T>
      <<<gridSize, blockSize>>>(m_array[n].data_d(), m_grid->extent(),
                                detail::Op_MinusConst<T>(value));
  return (*this);
}

template <typename T>
cu_vector_field<T>&
cu_vector_field<T>::subtractBy(const cu_vector_field<T>& field) {
  this->check_grid_extent(this->m_grid->extent(),
                          field.grid().extent());

  dim3 gridSize(8, 8, 8);
  dim3 blockSize(8, 8, 8);
  for (int i = 0; i < VECTOR_DIM; ++i) {
    Kernels::map_array_binary_op<T><<<gridSize, blockSize>>>(
        field.data(i).data_d(), m_array[i].data_d(), m_grid->extent(),
        detail::Op_MinusAssign<T>());
    // detail::map_multi_array(m_array[i].begin(),
    // field.data(i).begin(),
    //                         this -> m_grid -> extent(),
    //                         detail::Op_MinusAssign<T>());
  }
  return (*this);
}

template <typename T>
// template <int Order>
Vec3<T>
cu_vector_field<T>::interpolate(const Vec3<int>& c,
                                const Vec3<Pos_t>& rel_pos,
                                int order) const {
  Interpolator interp(order);
  // Vec3<int> c = m_grid -> mesh().getCell(cell);
  Vec3<int> lower = c - interp.radius();
  Vec3<int> upper = c + interp.support() - interp.radius();
  if (m_grid->dim() < 3) {
    lower[2] = upper[2] = c[2];
  }
  if (m_grid->dim() < 2) {
    lower[1] = upper[1] = c[1];
  }

  Vec3<T> result{0.0, 0.0, 0.0};
  for (int k = lower[2]; k <= upper[2]; k++) {
    for (int j = lower[1]; j <= upper[1]; j++) {
      for (int i = lower[0]; i <= upper[0]; i++) {
        if (m_grid->dim() < 3) {
          result[0] +=
              m_array[0](i, j, k) *
              interp.interp_cell(rel_pos[0], c[0], i, m_stagger[0][0]) *
              interp.interp_cell(rel_pos[1], c[1], j, m_stagger[0][1]);
          // * (normalize ? m_grid -> norm(0, i, j, k) : 1.0);
          // result[1] += m_array[1](i, j, k) *
          // interp.interp_cell(rel_pos[0], c[0], i, m_stagger[1][0])
          //              * interp.interp_cell(rel_pos[1], c[1], j,
          //              m_stagger[1][1]);
          // * (normalize ? m_grid -> norm(1, i, j, k) : 1.0);
          // result[2] += m_array[2](i, j, k) *
          // interp.interp_cell(rel_pos[0], c[0], i, m_stagger[2][0])
          //              * interp.interp_cell(rel_pos[1], c[1], j,
          //              m_stagger[2][1]);
          // * (normalize ? m_grid -> norm(2, i, j, k) : 1.0);
        } else {
          result[0] +=
              m_array[0](i, j, k) *
              interp.interp_weight(rel_pos, c, Vec3<int>(i, j, k),
                                   m_stagger[0]);
          // * (normalize ? m_grid -> norm(0, i, j, k) : 1.0);
          result[1] +=
              m_array[1](i, j, k) *
              interp.interp_weight(rel_pos, c, Vec3<int>(i, j, k),
                                   m_stagger[1]);
          // * (normalize ? m_grid -> norm(1, i, j, k) : 1.0);
          result[2] +=
              m_array[2](i, j, k) *
              interp.interp_weight(rel_pos, c, Vec3<int>(i, j, k),
                                   m_stagger[2]);
          // * (normalize ? m_grid -> norm(2, i, j, k) : 1.0);
        }
      }
    }
  }
  return result;
}

template <typename T>
void
cu_vector_field<T>::interpolate_to_center(self_type& result) {
  result.initialize();
  auto& mesh = m_grid->mesh();

  dim3 blockSize(16, 8, 8);
  dim3 gridSize(mesh.reduced_dim(0) / 16, mesh.reduced_dim(1) / 8,
                mesh.reduced_dim(2) / 8);

  Kernels::interp_to_center<2, 16, 8, 8><<<gridSize, blockSize>>>(
      result.ptr(0), result.ptr(1), result.ptr(2), m_array[0].data_d(),
      m_array[1].data_d(), m_array[2].data_d(), m_type);
  CudaCheckError();
}

template <typename T>
void
cu_vector_field<T>::interpolate_from_center(self_type& result,
                                            Scalar q) {
  result.initialize();
  auto& mesh = m_grid->mesh();

  if (mesh.dim() == 3) {
    dim3 blockSize(32, 8, 4);
    dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 8,
                  mesh.reduced_dim(2) / 4);

    Kernels::interp_from_center<2, 32, 8, 4><<<gridSize, blockSize>>>(
        result.ptr(0), result.ptr(1), result.ptr(2),
        m_array[0].data_d(), m_array[1].data_d(), m_array[2].data_d(),
        m_type);
    CudaCheckError();
  } else if (mesh.dim() == 2) {
    dim3 blockSize(32, 32);
    dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 32);

    Kernels::interp_from_center_2d<2, 32, 32><<<gridSize, blockSize>>>(
        result.ptr(0), result.ptr(1), result.ptr(2),
        m_array[0].data_d(), m_array[1].data_d(), m_array[2].data_d(),
        m_type, q);
    CudaCheckError();
  }
}

template <typename T>
void
cu_vector_field<T>::interpolate_from_center_add(self_type& result,
                                                Scalar q) {
  auto& mesh = m_grid->mesh();

  if (mesh.dim() == 3) {
    dim3 blockSize(32, 8, 4);
    dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 8,
                  mesh.reduced_dim(2) / 4);

    Kernels::interp_from_center<2, 32, 8, 4><<<gridSize, blockSize>>>(
        result.ptr(0), result.ptr(1), result.ptr(2),
        m_array[0].data_d(), m_array[1].data_d(), m_array[2].data_d(),
        m_type, q);
    CudaCheckError();
  } else if (mesh.dim() == 2) {
    dim3 blockSize(32, 32);
    dim3 gridSize(mesh.reduced_dim(0) / 32, mesh.reduced_dim(1) / 32);

    Kernels::interp_from_center_add_2d<2, 32, 32>
        <<<gridSize, blockSize>>>(result.ptr(0), result.ptr(1),
                                  result.ptr(2), m_array[0].data_d(),
                                  m_array[1].data_d(),
                                  m_array[2].data_d(), m_type, q);
    CudaCheckError();
  }
}

// template <typename T>
// void
// cu_vector_field<T>::recenter(self_type &output) const {
//   check_grid_extent(m_grid -> extent(), output.m_grid -> extent());
//   output.assign(0.0);
//   auto& mesh = m_grid -> mesh();
//   auto center = Vec3<Pos_t>(0.5, 0.5, 0.5);
//   for (int k = 0; k < mesh.dims[2]; k++) {
//     for (int j = 0; j < mesh.dims[1]; j++) {
//       for (int i = 0; i < mesh.dims[0]; i++) {
//         auto v = interpolate(Vec3<int>(i, j, k), center, 1);
//         output(0, i, j, k) = v.x;
//         output(1, i, j, k) = v.y;
//         output(2, i, j, k) = v.z;
//       }
//     }
//   }
// }

// template <typename T>
// void
// cu_vector_field<T>::normalize(FieldNormalization normalization) {
//   // Nothing to do if the normalization is already correct
//   if (normalization == m_normalization) return;
//   // FIXME: This does not seem to respect stagger
//   auto& mesh = m_grid -> mesh();
//   for (int k = 0; k < mesh.dims[2]; k++) {
//     int idx_k = k * mesh.idx_increment(2);
//     for (int j = 0; j < mesh.dims[1]; j++) {
//       int idx_j = j * mesh.idx_increment(1);
//       for (int i = 0; i < mesh.dims[0]; i++) {
//         int idx = i + idx_j + idx_k;
//         for (int comp = 0; comp < 3; comp++) {
//           // First convert to coord
//           if (m_normalization == FieldNormalization::physical) {
//             m_array[comp][idx] /= std::sqrt(m_grid -> metric(comp,
//             comp, idx));
//           } else if (m_normalization == FieldNormalization::volume) {
//             // FIXME: Careful about divide by zero!!
//             m_array[comp][idx] /= m_grid -> det(comp, idx);
//           }
//           if (normalization == FieldNormalization::physical) {
//             m_array[comp][idx] *= std::sqrt(m_grid -> metric(comp,
//             comp, idx));
//           } else if (normalization == FieldNormalization::volume) {
//             m_array[comp][idx] *= m_grid -> det(comp, idx);
//           }
//         }
//       }
//     }
//   }
//   m_normalization = normalization;
// }

// template <typename T>
// cu_vector_field<T>&
// cu_vector_field<T>::convertToFlux() {
//   for (int k = 0; k < m_grid -> mesh().dims[2]; k++) {
//     for (int j = 0; j < m_grid -> mesh().dims[1]; j++) {
//       for (int i = 0; i < m_grid -> mesh().dims[0]; i++) {
//         for (int n = 0; n < 3; n++) {
//           // double area = m_grid -> face_area(n, i, j, k);
//           // if (normalize)
//           //   m_array[n](i, j, k) /= m_grid -> norm(n, i, j, k);
//           m_array[n](i, j, k) *= m_grid -> face_area(n, i, j, k);
//         }
//       }
//     }
//   }
//   return *this;
// }

// template <typename T>
// cu_vector_field<T>&
// cu_vector_field<T>::convertFromFlux() {
//   for (int k = 0; k < m_grid -> mesh().dims[2]; k++) {
//     for (int j = 0; j < m_grid -> mesh().dims[1]; j++) {
//       for (int i = 0; i < m_grid -> mesh().dims[0]; i++) {
//         for (int n = 0; n < 3; n++) {
//           double area = m_grid -> face_area(n, i, j, k);
//           // In case the area is zero, do nothing
//           // FIXME: 1.0e-6 is magic number!
//           if (std::abs(area) > 1.0e-6)
//             m_array[n](i, j, k) /= area;
//           // if (normalize)
//           //   m_array[n](i, j, k) *= m_grid -> norm(n, i, j, k);
//         }
//       }
//     }
//   }
//   return *this;
// }

template <typename T>
void
cu_vector_field<T>::set_field_type(Aperture::FieldType type) {
  m_type = type;
  // TODO: If less than 3D, some components do not need to be staggered
  if (type == FieldType::E) {
    set_stagger(0, Stagger(0b110));
    set_stagger(1, Stagger(0b101));
    set_stagger(2, Stagger(0b011));
  } else if (type == FieldType::B) {
    set_stagger(0, Stagger(0b001));
    set_stagger(1, Stagger(0b010));
    set_stagger(2, Stagger(0b100));
  }
}

template <typename T>
std::array<Stagger, VECTOR_DIM>
cu_vector_field<T>::stagger_dual() const {
  auto stagger = m_stagger;
  for (unsigned int i = 0; i < stagger.size(); i++) {
    // Only flip those directions that are inside the grid dimension
    if (i < m_grid->dim()) {
      stagger[i].flip(i);
    }
  }
  return stagger;
}

////////////////////////////////////////////////////////////////////////////////
//  Explicit instantiations
////////////////////////////////////////////////////////////////////////////////

template class cu_scalar_field<double>;
template class cu_scalar_field<float>;

template class cu_vector_field<double>;
template class cu_vector_field<float>;

}  // namespace Aperture