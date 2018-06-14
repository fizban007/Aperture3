#include "data/fields.h"
#include "data/detail/multi_array_utils.hpp"
#include "algorithms/interpolation.h"

namespace Aperture {

void FieldBase::check_grid_extent(const Extent& ext1, const Extent& ext2) const {
  if (ext1 != ext2) throw std::invalid_argument("Field grids don't match!");
}

////////////////////////////////////////////////////////////////////////////////
//  Scalar Field Implementation
////////////////////////////////////////////////////////////////////////////////

// template <typename T>
// ScalarField<T>::ScalarField()
//     : FieldBase(), m_array() {}

template <typename T>
ScalarField<T>::ScalarField(const grid_type& grid, Stagger stagger)
    : FieldBase(grid), m_array(grid.extent()), m_stagger(stagger) {
   // std::cout << grid.extent() << std::endl;
}

template <typename T>
ScalarField<T>::ScalarField(const self_type& field)
    : FieldBase(*field.m_grid), m_array(field.m_array)
    , m_stagger(field.m_stagger) {}

template <typename T>
ScalarField<T>::ScalarField(self_type&& field)
    : FieldBase(*field.m_grid), m_array(std::move(field.m_array))
    , m_stagger(field.m_stagger) {}

template <typename T>
ScalarField<T>::~ScalarField() {}

template <typename T>
void ScalarField<T>::initialize() {
  // Assign the field to zero, whatever 0 corresponds to for type #T
  m_array.assign(static_cast<T>(0));
}

template <typename T>
void ScalarField<T>::assign(data_type value) {
  // Assign a uniform value to the array
  m_array.assign(value);
}

template <typename T>
void ScalarField<T>::copyFrom(const self_type& field) {
  /// We can copy as long as the extents are the same
  this -> check_grid_extent(this -> m_grid -> extent(), field.grid().extent());

  m_array.copyFrom(field.m_array);
}

template <typename T>
ScalarField<T>& ScalarField<T>::operator = ( const self_type& field){
    this -> m_grid = field.m_grid;
    this -> m_grid_size = field.m_grid_size;
    m_array = field.m_array;
    return (*this);
}

template <typename T>
ScalarField<T>& ScalarField<T>::operator = ( self_type&& field){
    this -> m_grid = field.m_grid;
    this -> m_grid_size = field.m_grid_size;
    m_array = std::move(field.m_array);
    return (*this);
}

template <typename T>
void ScalarField<T>::resize (const Grid& grid) {
  this -> m_grid = &grid;
  this -> m_grid_size = grid.size();
  m_array.resize(grid.extent());
}

template <typename T>
ScalarField<T>& ScalarField<T>::multiplyBy(data_type value) {
  // detail::map_multi_array(m_array.begin(), this -> m_grid -> extent(), detail::Op_MultConst<T>(value));
  Kernels::map_array_unary_op<<<256, 256>>>(m_array.data(), m_grid->extent(), detail::Op_MultConst<T>(value));
  return (*this);
}

template <typename T>
ScalarField<T>& ScalarField<T>::multiplyBy(
    const ScalarField<T>& field) {
  this -> check_grid_extent(this -> m_grid -> extent(), field.grid().extent());

  // detail::map_multi_array(m_array.begin(), field.data().begin(), this -> m_grid -> extent(),
  //                         detail::Op_MultAssign<T>());
  Kernels::map_array_binary_op<<<256, 256>>>(field.ptr(), m_array.data(), m_grid->extent(),
                                  detail::Op_MultAssign<T>());
  return (*this);
}

template <typename T>
ScalarField<T>& ScalarField<T>::addBy(data_type value) {
  // detail::map_multi_array(m_array.begin(), this -> m_grid -> extent(), detail::Op_PlusConst<T>(value));
  Kernels::map_array_unary_op<<<256, 256>>>(m_array.data(), m_grid->extent(), detail::Op_PlusConst<T>(value));
  return (*this);
}

template <typename T>
ScalarField<T>& ScalarField<T>::addBy(
    const ScalarField<T>& field) {
  this -> check_grid_extent(this -> m_grid -> extent(), field.grid().extent());

  // detail::map_multi_array(m_array.begin(), field.data().begin(), this -> m_grid -> extent(),
  //                         detail::Op_PlusAssign<T>());
  Kernels::map_array_binary_op<<<256, 256>>>(field.ptr(), m_array.data(), m_grid->extent(),
                                  detail::Op_PlusAssign<T>());
  return (*this);
}

template <typename T>
ScalarField<T>& ScalarField<T>::subtractBy(data_type value) {
  // detail::map_multi_array(m_array.begin(), this -> m_grid -> extent(), detail::Op_MinusConst<T>(value));
  Kernels::map_array_unary_op<<<256, 256>>>(m_array.data(), m_grid->extent(), detail::Op_MinusConst<T>(value));
  return (*this);
}

template <typename T>
ScalarField<T>& ScalarField<T>::subtractBy(const ScalarField<T> &field) {
  this -> check_grid_extent(this -> m_grid -> extent(), field.grid().extent());

  // detail::map_multi_array(m_array.begin(), field.data().begin(), this -> m_grid -> extent(),
  //                         detail::Op_MinusAssign<T>());
  Kernels::map_array_binary_op<<<256, 256>>>(field.ptr(), m_array.data(), m_grid->extent(),
                                  detail::Op_MinusAssign<T>());
  return (*this);

}

template <typename T>
// template <int Order>
T ScalarField<T>::interpolate(const Vec3<int>& c, const Vec3<Pos_t>& rel_pos, int order) const {
  Interpolator interp(order);
  // Vec3<int> c = m_grid -> mesh().getCell(cell);
  Vec3<int> lower = c - interp.radius();
  Vec3<int> upper = c + interp.support() - interp.radius();
  if (m_grid -> dim() < 3) {
    lower[2] = upper[2] = c[2];
  }
  T result {};
  for (int k = lower[2]; k <= upper[2]; k++) {
    for (int j = lower[1]; j <= upper[1]; j++) {
      for (int i = lower[0]; i <= upper[0]; i++) {
        if (m_grid -> dim() < 3) {
          result += m_array(i, j, k) * interp.interp_cell(rel_pos[0], c[0], i, m_stagger[0]) * interp.interp_cell(rel_pos[1], c[1], j, m_stagger[1]);
        } else {
          result += m_array(i, j, k) * interp.interp_weight(rel_pos, c, Vec3<int>(i, j, k), m_stagger);
        }
      }
    }
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////
//  Vector Field Implementation
////////////////////////////////////////////////////////////////////////////////

// template <typename T>
// VectorField<T>::VectorField()
//     : FieldBase(), m_array() {
//   for (int i = 0; i < VECTOR_DIM; ++i) {
//     // Default initialize to face-centered
//     m_stagger[i] = Stagger("000");
//     m_stagger[i][i] = true;
//   }
// }

template <typename T>
VectorField<T>::VectorField(const grid_type& grid)
    : FieldBase(grid) {
  for (int i = 0; i < VECTOR_DIM; ++i) {
    m_array[i] = array_type(grid.extent());
    // Default initialize to face-centered
    m_stagger[i] = Stagger();
    m_stagger[i].set_bit(i, true);
  }
}

template <typename T>
VectorField<T>::VectorField(const self_type& field)
    : FieldBase(*field.m_grid), m_array(field.m_array)
    , m_stagger(field.m_stagger) {}

template <typename T>
VectorField<T>::VectorField(self_type&& field)
    : FieldBase(*field.m_grid), m_array(std::move(field.m_array))
    , m_stagger(field.m_stagger) {}

template <typename T>
VectorField<T>::~VectorField() {}

template <typename T>
VectorField<T>& VectorField<T>::operator= (const self_type& other) {
    this -> m_grid = other.m_grid;
    this -> m_grid_size = other.m_grid_size;
    m_array = other.m_array;
    return (*this);
}

template <typename T>
VectorField<T>& VectorField<T>::operator= ( self_type&& other) {
    this -> m_grid = other.m_grid;
    this -> m_grid_size = other.m_grid_size;
    m_array = std::move(other.m_array);
    return (*this);
}

template <typename T>
void VectorField<T>::initialize() {
  for (int i = 0; i < VECTOR_DIM; ++i) {
    m_array[i].assign(static_cast<T>(0));
  }
}

// template <typename T>
// void vector_field<T>::initialize(const initial_condition& ic, FieldType type) {
//   for (int component = 0; component < 3; component ++) {
//     // This way vector field is always defined in the center of the cell
//     // face, staggered in the direction of the component
//     for (int k = 0; k < m_grid -> extent().depth(); ++k) {
//       double x3 = m_grid -> mesh().pos(2, k, component == 2 ? 1 : 0);
//       for (int j = 0; j < m_grid -> extent().height(); ++j) {
//         double x2 = m_grid -> mesh().pos(1, j, component == 1 ? 1 : 0);
//         for (int i = 0; i < m_grid -> extent().width(); ++i) {
//           double x1 = m_grid -> mesh().pos(0, i, component == 0 ? 1 : 0);
//           // Stagger is automatically taken care of by the grid
//           if (type == FieldType::E)
//             m_array[component](i, j, k) = ic.E(component, x1, x2, x3);
//           else if (type == FieldType::B)
//             m_array[component](i, j, k) = ic.B(component, x1, x2, x3);
//           if (type == FieldType::B && i == 20 && j == m_grid -> mesh().dims[1] - m_grid -> mesh().guard[1] - 1)
//             std::cout << Vec3<double>(x1, x2, x3) << std::endl;
//         }
//       }
//     }
//   }
// }

template <typename T>
void VectorField<T>::assign(data_type value, int n) {
  m_array[n].assign(value);
}

template <typename T>
void VectorField<T>::assign(data_type value) {
  for (int i = 0; i < VECTOR_DIM; i++) {
    m_array[i].assign(value);
  }
}

template <typename T>
void VectorField<T>::copyFrom(const self_type& field) {
  /// We can copy as long as the extents are the same
  this -> check_grid_extent(this -> m_grid -> extent(), field.grid().extent());

  for (int i = 0; i < VECTOR_DIM; ++i) {
    m_array[i].copyFrom(field.m_array[i]);
  }
}

template <typename T>
void VectorField<T>::resize (const Grid& grid) {
  this -> m_grid = &grid;
  this -> m_grid_size = grid.size();
  for (int i = 0; i < VECTOR_DIM; i++) {
    m_array[i].resize(grid.extent());
  }
}

template <typename T>
VectorField<T>& VectorField<T>::multiplyBy(data_type value) {
  for (int i = 0; i < VECTOR_DIM; ++i) {
    // detail::map_multi_array(m_array[i].begin(), this -> m_grid -> extent(),
    //                         detail::Op_MultConst<T>(value));
    Kernels::map_array_unary_op<<<256, 256>>>(m_array[i].data(),
                                                 m_grid->extent(), detail::Op_MultConst<T>(value));
  }
  return (*this);
}

template <typename T>
VectorField<T>& VectorField<T>::multiplyBy(const ScalarField<T>& field) {
  this -> check_grid_extent(this -> m_grid -> extent(), field.grid().extent());

  for (int i = 0; i < VECTOR_DIM; ++i) {
    Kernels::map_array_binary_op<<<256, 256>>>(field.ptr(), m_array[i].data(), m_grid->extent(),
                                  detail::Op_MultAssign<T>());
    // detail::map_multi_array(m_array[i].begin(), field.data().begin(),
    //                         this -> m_grid -> extent(), detail::Op_MultAssign<T>());
  }
  return (*this);
}

template <typename T>
VectorField<T>& VectorField<T>::addBy(data_type value, int n) {
  // detail::map_multi_array(m_array[n].begin(), this -> m_grid -> extent(),
  //                         detail::Op_PlusConst<T>(value));
  Kernels::map_array_unary_op<<<256, 256>>>(m_array[n].data(),
                                               m_grid->extent(), detail::Op_PlusConst<T>(value));
  return (*this);
}

template <typename T>
VectorField<T>& VectorField<T>::addBy(const VectorField<T>& field) {
  this -> check_grid_extent(this -> m_grid -> extent(), field.grid().extent());

  for (int i = 0; i < VECTOR_DIM; ++i) {
    // detail::map_multi_array(m_array[i].begin(), field.data(i).begin(),
    //                         this -> m_grid -> extent(), detail::Op_PlusAssign<T>());
    Kernels::map_array_binary_op<<<256, 256>>>(field.ptr(i), m_array[i].data(), m_grid->extent(),
                                  detail::Op_PlusAssign<T>());
  }
  return (*this);
}

template <typename T>
VectorField<T>& VectorField<T>::subtractBy(data_type value, int n) {
  // detail::map_multi_array(m_array[n].begin(), this -> m_grid -> extent(),
  //                         detail::Op_MinusConst<T>(value));
  Kernels::map_array_unary_op<<<256, 256>>>(m_array[n].data(),
                                               m_grid->extent(), detail::Op_MinusConst<T>(value));
  return (*this);
}

template <typename T>
VectorField<T>& VectorField<T>::subtractBy(const VectorField<T> &field) {
  this -> check_grid_extent(this -> m_grid -> extent(), field.grid().extent());

  for (int i = 0; i < VECTOR_DIM; ++i) {
    Kernels::map_array_binary_op<<<256, 256>>>(field.ptr(i), m_array[i].data(), m_grid->extent(),
                                  detail::Op_MinusAssign<T>());
    // detail::map_multi_array(m_array[i].begin(), field.data(i).begin(),
    //                         this -> m_grid -> extent(), detail::Op_MinusAssign<T>());
  }
  return (*this);
}

template <typename T>
// template <int Order>
Vec3<T>
VectorField<T>::interpolate(const Vec3<int>& c, const Vec3<Pos_t>& rel_pos, int order) const {
  Interpolator interp(order);
  // Vec3<int> c = m_grid -> mesh().getCell(cell);
  Vec3<int> lower = c - interp.radius();
  Vec3<int> upper = c + interp.support() - interp.radius();
  if (m_grid -> dim() < 3) {
    lower[2] = upper[2] = c[2];
  }
  if (m_grid -> dim() < 2) {
    lower[1] = upper[1] = c[1];
  }

  Vec3<T> result {0.0, 0.0, 0.0};
  for (int k = lower[2]; k <= upper[2]; k++) {
    for (int j = lower[1]; j <= upper[1]; j++) {
      for (int i = lower[0]; i <= upper[0]; i++) {
        if (m_grid -> dim() < 3) {
          result[0] += m_array[0](i, j, k) * interp.interp_cell(rel_pos[0], c[0], i, m_stagger[0][0])
                       * interp.interp_cell(rel_pos[1], c[1], j, m_stagger[0][1]);
          // * (normalize ? m_grid -> norm(0, i, j, k) : 1.0);
          // result[1] += m_array[1](i, j, k) * interp.interp_cell(rel_pos[0], c[0], i, m_stagger[1][0])
          //              * interp.interp_cell(rel_pos[1], c[1], j, m_stagger[1][1]);
          // * (normalize ? m_grid -> norm(1, i, j, k) : 1.0);
          // result[2] += m_array[2](i, j, k) * interp.interp_cell(rel_pos[0], c[0], i, m_stagger[2][0])
          //              * interp.interp_cell(rel_pos[1], c[1], j, m_stagger[2][1]);
          // * (normalize ? m_grid -> norm(2, i, j, k) : 1.0);
        } else {
          result[0] += m_array[0](i, j, k) * interp.interp_weight(rel_pos, c, Vec3<int>(i, j, k), m_stagger[0]);
          // * (normalize ? m_grid -> norm(0, i, j, k) : 1.0);
          result[1] += m_array[1](i, j, k) * interp.interp_weight(rel_pos, c, Vec3<int>(i, j, k), m_stagger[1]);
          // * (normalize ? m_grid -> norm(1, i, j, k) : 1.0);
          result[2] += m_array[2](i, j, k) * interp.interp_weight(rel_pos, c, Vec3<int>(i, j, k), m_stagger[2]);
          // * (normalize ? m_grid -> norm(2, i, j, k) : 1.0);
        }
      }
    }
  }
  return result;
}

template <typename T>
void
VectorField<T>::recenter(self_type &output) const {
  check_grid_extent(m_grid -> extent(), output.m_grid -> extent());
  output.assign(0.0);
  auto& mesh = m_grid -> mesh();
  auto center = Vec3<Pos_t>(0.5, 0.5, 0.5);
  for (int k = 0; k < mesh.dims[2]; k++) {
    for (int j = 0; j < mesh.dims[1]; j++) {
      for (int i = 0; i < mesh.dims[0]; i++) {
        auto v = interpolate(Vec3<int>(i, j, k), center, 1);
        output(0, i, j, k) = v.x;
        output(1, i, j, k) = v.y;
        output(2, i, j, k) = v.z;
      }
    }
  }
}

// template <typename T>
// void
// VectorField<T>::normalize(FieldNormalization normalization) {
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
//             m_array[comp][idx] /= std::sqrt(m_grid -> metric(comp, comp, idx));
//           } else if (m_normalization == FieldNormalization::volume) {
//             // FIXME: Careful about divide by zero!!
//             m_array[comp][idx] /= m_grid -> det(comp, idx);
//           }
//           if (normalization == FieldNormalization::physical) {
//             m_array[comp][idx] *= std::sqrt(m_grid -> metric(comp, comp, idx));
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
// VectorField<T>&
// VectorField<T>::convertToFlux() {
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
// VectorField<T>&
// VectorField<T>::convertFromFlux() {
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
VectorField<T>::set_field_type(Aperture::FieldType type) {
  if (type == FieldType::E) {
    set_stagger(0, Stagger(0b001));
    set_stagger(1, Stagger(0b010));
    set_stagger(2, Stagger(0b100));
  } else if (type == FieldType::B) {
    set_stagger(0, Stagger(0b110));
    set_stagger(1, Stagger(0b101));
    set_stagger(2, Stagger(0b011));
  }
}

template <typename T>
std::array<Stagger, VECTOR_DIM>
VectorField<T>::stagger_dual() const {
  auto stagger = m_stagger;
  for (unsigned int i = 0; i < stagger.size(); i++) {
    // Only flip those directions that are inside the grid dimension
    if (i < m_grid -> dim()) {
      stagger[i].flip(i);
    }
  }
  return stagger;
}

////////////////////////////////////////////////////////////////////////////////
//  Explicit instantiations
////////////////////////////////////////////////////////////////////////////////

template class ScalarField<double>;
template class ScalarField<float>;

template class VectorField<double>;
template class VectorField<float>;

}
